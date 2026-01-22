"""
WEEX Exchange Execution Client
==============================

Handles order placement and position management for WEEX Futures.

Endpoints used:
- POST /capi/v2/order/placeOrder - Place new orders with TP/SL
- POST /capi/v2/order/closePositions - Close existing positions
"""
import time
import hmac
import hashlib
import base64
import json
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiohttp.resolver import AsyncResolver
import structlog
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.settings import settings

logger = structlog.get_logger(__name__)

# =============================================================================
# SYMBOL MAPPING
# =============================================================================

SYMBOL_MAP: Dict[str, str] = {
    "BTCUSDT": "cmt_btcusdt",
    "ETHUSDT": "cmt_ethusdt",
    "SOLUSDT": "cmt_solusdt",
    "DOGEUSDT": "cmt_dogeusdt",
    "XRPUSDT": "cmt_xrpusdt",
    "ADAUSDT": "cmt_adausdt",
    "BNBUSDT": "cmt_bnbusdt",
    "LTCUSDT": "cmt_ltcusdt",
}

# Step sizes for each symbol (minimum order increment)
STEP_SIZES: Dict[str, float] = {
    "BTCUSDT": 0.001,    # 0.001 BTC
    "ETHUSDT": 0.01,     # 0.01 ETH
    "SOLUSDT": 0.1,      # 0.1 SOL
    "DOGEUSDT": 10.0,    # 10 DOGE
    "XRPUSDT": 10.0,      # 1 XRP
    "ADAUSDT": 10.0,     # 10 ADA (per exchange requirement)
    "BNBUSDT": 0.01,     # 0.01 BNB
    "LTCUSDT": 0.01,     # 0.01 LTC
}

# Approximate prices for size calculation (updated dynamically)
PRICE_CACHE: Dict[str, float] = {
    "BTCUSDT": 95000.0,
    "ETHUSDT": 3500.0,
    "SOLUSDT": 200.0,
    "DOGEUSDT": 0.35,
    "XRPUSDT": 2.5,
    "ADAUSDT": 1.0,
    "BNBUSDT": 700.0,
    "LTCUSDT": 100.0,
}


class OrderType(Enum):
    OPEN_LONG = "1"
    OPEN_SHORT = "2"
    CLOSE_LONG = "3"
    CLOSE_SHORT = "4"


@dataclass
class OrderResult:
    """Result of an order placement"""
    success: bool
    order_id: Optional[str] = None
    client_oid: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class Position:
    """Tracked position state"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    size: float
    order_id: str
    client_oid: str
    stop_price: float
    target_price: float
    opened_at: float = field(default_factory=time.time)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.direction == "LONG":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


class WeexClient:
    """
    WEEX Futures API Client
    
    Handles authenticated requests to WEEX exchange for:
    - Placing orders with take-profit and stop-loss
    - Closing positions
    - Position tracking
    """
    
    BASE_URL = "https://api-contract.weex.com"
    
    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        passphrase: str = None,
        equity: float = 1000.0,
        dry_run: bool = False,
    ):
        self.api_key = api_key or settings.WEEX_API_KEY
        self.secret_key = secret_key or settings.WEEX_SECRET_KEY
        self.passphrase = passphrase or settings.WEEX_PASSPHRASE
        self.equity = equity
        self.dry_run = dry_run
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        
        # Stats
        self.total_orders_placed = 0
        self.total_orders_failed = 0
        self.total_positions_closed = 0
        
        # Session
        self._session: Optional[aiohttp.ClientSession] = None
        
        if not all([self.api_key, self.secret_key, self.passphrase]):
            logger.warning("weex_credentials_missing", 
                         message="WEEX API credentials not configured - orders will fail")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            try:
                resolver = AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4", "1.1.1.1"])
            except Exception:
                resolver = None
 
            connector = TCPConnector(
                ttl_dns_cache=300,
                limit=100,
                limit_per_host=30,
                force_close=False,
                enable_cleanup_closed=True,
                family=0,
                resolver=resolver,
                use_dns_cache=True,
            )
            timeout = ClientTimeout(total=30, connect=10, sock_connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def check_connectivity(self) -> Tuple[bool, str]:
        """
        Check WEEX API connectivity by fetching all positions.
        
        Returns:
            Tuple of (success, message)
        """
        if self.dry_run:
            logger.info("weex_connectivity_check", status="skipped", reason="dry_run_mode")
            return True, "Dry-run mode - connectivity check skipped"
        
        if not all([self.api_key, self.secret_key, self.passphrase]):
            return False, "Missing WEEX API credentials"
        
        try:
            # GET /capi/v2/account/position/allPosition
            timestamp = str(int(time.time() * 1000))
            endpoint = "/capi/v2/account/position/allPosition"
            query_string = ""
            
            signature = self._generate_signature(
                timestamp, "GET", endpoint, query_string, ""
            )
            headers = self._get_headers(timestamp, signature)
            
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}{endpoint}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                text = await response.text()
                
                if response.status == 200:
                    try:
                        data = json.loads(text)
                        if isinstance(data, list):
                            logger.info(
                                "weex_connectivity_check",
                                status="success",
                                positions_count=len(data),
                            )
                            return True, f"Connected to WEEX. Active positions: {len(data)}"
                        elif isinstance(data, dict):
                            if data.get("code") == "00000":
                                positions = data.get("data", [])
                                logger.info(
                                    "weex_connectivity_check",
                                    status="success",
                                    positions_count=len(positions) if positions else 0,
                                )
                                return True, f"Connected to WEEX. Active positions: {len(positions) if positions else 0}"
                            else:
                                return False, f"WEEX API error: {data.get('msg', 'Unknown')}"
                        else:
                            return True, "Connected to WEEX"
                    except json.JSONDecodeError:
                        return False, f"Invalid JSON response: {text[:100]}"
                else:
                    logger.error(
                        "weex_connectivity_check",
                        status="failed",
                        http_status=response.status,
                    )
                    return False, f"HTTP {response.status}: {text[:100]}"
                    
        except asyncio.TimeoutError:
            return False, "Connection timeout - check network/VPN"
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str,
        body: str
    ) -> str:
        """Generate HMAC-SHA256 signature for request"""
        message = timestamp + method.upper() + request_path + query_string + body
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _get_headers(self, timestamp: str, signature: str) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
            "User-Agent": "HydraV3/1.0"
        }
    
    async def _post(
        self,
        endpoint: str,
        body: Dict[str, Any],
        query_string: str = ""
    ) -> Tuple[bool, Dict]:
        """Make authenticated POST request"""
        timestamp = str(int(time.time() * 1000))
        body_str = json.dumps(body)
        signature = self._generate_signature(
            timestamp, "POST", endpoint, query_string, body_str
        )
        headers = self._get_headers(timestamp, signature)
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.BASE_URL}{endpoint}",
                headers=headers,
                data=body_str,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                text = await response.text()
                
                if response.status == 200:
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        return False, {"error": "Invalid JSON response", "text": text}

                    # WEEX typically returns {"code":"00000", "data":{...}, "msg":"..."}
                    # Some endpoints may return a dict without a top-level code on success.
                    if isinstance(data, dict) and "code" in data:
                        return data.get("code") == "00000", data
                    return True, data
                else:
                    logger.error("weex_request_failed",
                               status=response.status,
                               endpoint=endpoint,
                               response=text[:500])
                    return False, {
                        "error": f"HTTP {response.status}",
                        "status": response.status,
                        "endpoint": endpoint,
                        "text": text,
                    }
                    
        except asyncio.TimeoutError:
            logger.error("weex_request_timeout", endpoint=endpoint)
            return False, {"error": "Request timeout"}
        except Exception as e:
            logger.error("weex_request_error", endpoint=endpoint, error=str(e))
            return False, {"error": str(e)}
    
    def _calculate_size(
        self,
        symbol: str,
        notional_usd: float,
        current_price: float = None
    ) -> str:
        """
        Calculate order size based on notional USD value
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            notional_usd: USD value of position
            current_price: Current price (uses cache if not provided)
            
        Returns:
            Size string formatted for WEEX API
        """
        price = current_price or PRICE_CACHE.get(symbol, 1.0)
        step_size = STEP_SIZES.get(symbol, 0.001)
        
        # Calculate raw size
        raw_size = notional_usd / price
        
        # Round to step size
        steps = int(raw_size / step_size)
        size = steps * step_size
        
        # Format appropriately
        if step_size >= 1:
            return str(int(size))
        else:
            decimals = len(str(step_size).split('.')[-1])
            return f"{size:.{decimals}f}"
    
    def update_price(self, symbol: str, price: float):
        """Update price cache for size calculation"""
        PRICE_CACHE[symbol] = price
    
    async def place_order(
        self,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        size_usd: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
        client_oid: str = None,
    ) -> OrderResult:
        """
        Place a new order with take-profit and stop-loss
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            direction: "LONG" or "SHORT"
            size_usd: Position size in USD
            entry_price: Current market price (for size calculation)
            stop_price: Stop-loss price
            target_price: Take-profit price
            client_oid: Custom order ID (generated if not provided)
            
        Returns:
            OrderResult with success status and order details
        """
        # Map symbol
        weex_symbol = SYMBOL_MAP.get(symbol)
        if not weex_symbol:
            return OrderResult(
                success=False,
                error_message=f"Unknown symbol: {symbol}"
            )
        
        # Update price cache
        self.update_price(symbol, entry_price)
        
        # Calculate size
        size = self._calculate_size(symbol, size_usd, entry_price)
        
        # Determine order type
        order_type = OrderType.OPEN_LONG if direction == "LONG" else OrderType.OPEN_SHORT
        
        # Generate client order ID
        if client_oid is None:
            client_oid = f"hydra_{int(time.time() * 1000000)}"
        
        # Build order body
        body = {
            "symbol": weex_symbol,
            "client_oid": client_oid,
            "size": size,
            "type": order_type.value,
            "order_type": "0",       # Normal order
            "match_price": "1",      # Market price
            "price": "",             # Empty for market orders
            "presetTakeProfitPrice": str(round(target_price, 1)),
            "presetStopLossPrice": str(round(stop_price, 1)),
        }
        
        logger.info("weex_place_order",
                   symbol=symbol,
                   direction=direction,
                   size=size,
                   size_usd=size_usd,
                   entry_price=entry_price,
                   stop_price=stop_price,
                   target_price=target_price,
                   client_oid=client_oid,
                   dry_run=self.dry_run)
        
        if self.dry_run:
            # Simulate successful order in dry run mode
            fake_order_id = f"dry_{int(time.time() * 1000)}"
            self.total_orders_placed += 1
            
            # Track position
            self.positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                size=float(size),
                order_id=fake_order_id,
                client_oid=client_oid,
                stop_price=stop_price,
                target_price=target_price,
            )
            
            return OrderResult(
                success=True,
                order_id=fake_order_id,
                client_oid=client_oid,
                raw_response={"dry_run": True}
            )
        
        # Make actual API call
        success, response = await self._post("/capi/v2/order/placeOrder", body)
        
        # Log full response for debugging
        logger.info("weex_order_response", 
                   symbol=symbol,
                   success=success,
                   response=response)
        
        if success and response.get("code") == "00000":
            data = response.get("data", {})
            order_id = data.get("order_id") or data.get("orderId")
            
            self.total_orders_placed += 1
            
            # Track position
            self.positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                size=float(size),
                order_id=order_id,
                client_oid=client_oid,
                stop_price=stop_price,
                target_price=target_price,
            )
            
            logger.info("weex_order_placed",
                       symbol=symbol,
                       order_id=order_id,
                       client_oid=client_oid)
            
            return OrderResult(
                success=True,
                order_id=order_id,
                client_oid=client_oid,
                raw_response=response
            )
        
        # Some WEEX responses may omit the top-level code but still return an order id.
        if success:
            data = response.get("data") if isinstance(response, dict) else None
            if isinstance(data, dict):
                order_id = data.get("order_id") or data.get("orderId")
            else:
                order_id = response.get("order_id") if isinstance(response, dict) else None
                order_id = order_id or (response.get("orderId") if isinstance(response, dict) else None)
 
            if order_id:
                self.total_orders_placed += 1
                self.positions[symbol] = Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=float(size),
                    order_id=order_id,
                    client_oid=client_oid,
                    stop_price=stop_price,
                    target_price=target_price,
                )
                logger.info("weex_order_placed",
                           symbol=symbol,
                           order_id=order_id,
                           client_oid=client_oid)
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    client_oid=client_oid,
                    raw_response=response,
                )
        else:
            self.total_orders_failed += 1
            error_msg = response.get("msg") or response.get("error") or "Unknown error"
            error_code = response.get("code")
            
            logger.error("weex_order_failed",
                        symbol=symbol,
                        error_code=error_code,
                        error_message=error_msg)
            
            return OrderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                raw_response=response
            )
    
    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close an existing position
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            OrderResult with success status
        """
        # Map symbol
        weex_symbol = SYMBOL_MAP.get(symbol)
        if not weex_symbol:
            return OrderResult(
                success=False,
                error_message=f"Unknown symbol: {symbol}"
            )
        
        # Check if we have a tracked position
        position = self.positions.get(symbol)
        
        logger.info("weex_close_position",
                   symbol=symbol,
                   has_tracked_position=position is not None,
                   dry_run=self.dry_run)
        
        if self.dry_run:
            if position:
                del self.positions[symbol]
            self.total_positions_closed += 1
            return OrderResult(
                success=True,
                raw_response={"dry_run": True}
            )
        
        # Make API call
        body = {"symbol": weex_symbol}
        success, response = await self._post("/capi/v2/order/closePositions", body)
        
        if success and response.get("code") == "00000":
            # Remove from tracked positions
            if symbol in self.positions:
                del self.positions[symbol]
            
            self.total_positions_closed += 1
            
            logger.info("weex_position_closed", symbol=symbol)
            
            return OrderResult(
                success=True,
                raw_response=response
            )
        else:
            error_msg = response.get("msg") or response.get("error") or "Unknown error"
            error_code = response.get("code")
            
            logger.error("weex_close_failed",
                        symbol=symbol,
                        error_code=error_code,
                        error_message=error_msg)
            
            return OrderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                raw_response=response
            )
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for symbol"""
        return symbol in self.positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position should be closed based on TP/SL
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            "TP" if take-profit hit, "SL" if stop-loss hit, None otherwise
        """
        position = self.positions.get(symbol)
        if not position:
            return None
        
        if position.direction == "LONG":
            if current_price >= position.target_price:
                return "TP"
            if current_price <= position.stop_price:
                return "SL"
        else:  # SHORT
            if current_price <= position.target_price:
                return "TP"
            if current_price >= position.stop_price:
                return "SL"
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_orders_placed": self.total_orders_placed,
            "total_orders_failed": self.total_orders_failed,
            "total_positions_closed": self.total_positions_closed,
            "active_positions": len(self.positions),
            "position_symbols": list(self.positions.keys()),
            "dry_run": self.dry_run,
        }
    
    async def upload_ai_log(
        self,
        order_id: str,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        explanation: str,
    ) -> Tuple[bool, Dict]:
        """
        Upload AI decision log to WEEX for competition tracking.
        
        Args:
            order_id: The order ID from place_order
            stage: Processing stage (e.g., "Signal Detection & Quantitative Gating")
            model: Model name (e.g., "Hybrid-Quant Signal Engine v3")
            input_data: Input features dict
            output_data: Output/prediction dict
            explanation: Human-readable explanation
            
        Returns:
            Tuple of (success, response)
        """
        if self.dry_run:
            logger.info("ai_log_upload_skipped", reason="dry_run", order_id=order_id)
            return True, {"dry_run": True}
        
        body = {
            "orderId": order_id,
            "stage": stage,
            "model": model,
            "input": input_data,
            "output": output_data,
            "explanation": explanation,
        }
        
        success, response = await self._post("/capi/v2/order/uploadAiLog", body)
        
        if success:
            logger.info("ai_log_uploaded", order_id=order_id, stage=stage)
        else:
            logger.warning(
                "ai_log_upload_failed",
                order_id=order_id,
                endpoint="/capi/v2/order/uploadAiLog",
                error=response,
            )
        
        return success, response
