#!/usr/bin/env python3
"""
Upload AI Logs to WEEX API
==========================

Reads ai_logs.json and uploads each AI log to WEEX using the same API
as the live trading system.
"""

import asyncio
import json
import sys
import os
import time
import hmac
import hashlib
import base64
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiohttp.resolver import AsyncResolver

load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import settings
import structlog

logger = structlog.get_logger(__name__)

api_key = os.getenv("WEEX_API_KEY")
secret_key = os.getenv("WEEX_SECRET_KEY")
access_passphrase = os.getenv("WEEX_PASSPHRASE")
def _generate_signature(timestamp: str, method: str, endpoint: str, query: str, body: str) -> str:
    """Generate WEEX API signature"""
    message = timestamp + method + endpoint + query + body
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')


def _get_headers(timestamp: str, signature: str) -> dict:
    """Get request headers with authentication"""
    return {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US",
        "User-Agent": "HydraV3/1.0"
    }


async def upload_ai_logs():
    """Upload all AI logs from ai_logs.json to WEEX"""
    
    # Create session with custom DNS resolvers
    try:
        resolver = AsyncResolver(nameservers=['8.8.8.8', '8.8.4.4', '1.1.1.1'])
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
        use_dns_cache=True
    )
    
    timeout = ClientTimeout(total=30, connect=10, sock_connect=10)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            try:
                # Read AI logs
                ai_logs_path = Path(__file__).parent / "ai_logs.json"
                if not ai_logs_path.exists():
                    logger.error("ai_logs_file_not_found", path=str(ai_logs_path))
                    return
                
                with open(ai_logs_path, 'r') as f:
                    ai_logs = json.load(f)
                
                if not ai_logs:
                    logger.info("no_ai_logs_found")
                    return
                
                logger.info("uploading_ai_logs", count=len(ai_logs))
                
                # Upload each AI log
                for i, ai_log in enumerate(ai_logs, 1):
                    try:
                        # Extract required fields
                        order_id = ai_log["order_id"]
                        stage = ai_log["stage"]
                        model = ai_log["model"]
                        input_data = ai_log["input"]
                        output_data = ai_log["output"]
                        explanation = ai_log["explanation"]
                        
                        logger.info(
                            "uploading_ai_log",
                            index=i,
                            total=len(ai_logs),
                            order_id=order_id,
                            symbol=ai_log.get("symbol", "UNKNOWN")
                        )
                        
                        # Prepare request
                        endpoint = "/capi/v2/order/uploadAiLog"
                        body = {
                            "orderId": order_id,
                            "stage": stage,
                            "model": model,
                            "input": input_data,
                            "output": output_data,
                            "explanation": explanation,
                        }
                        
                        # Generate signature
                        timestamp = str(int(time.time() * 1000))
                        body_str = json.dumps(body)
                        signature = _generate_signature(timestamp, "POST", endpoint, "", body_str)
                        headers = _get_headers(timestamp, signature)
                        
                        # Make request
                        async with session.post(
                            f"https://api-contract.weex.com{endpoint}",
                            headers=headers,
                            data=body_str
                        ) as response:
                            text = await response.text()
                            
                            if response.status == 200:
                                logger.info(
                                    "ai_log_upload_success",
                                    order_id=order_id,
                                    response=text[:100]
                                )
                            else:
                                logger.error(
                                    "ai_log_upload_failed",
                                    order_id=order_id,
                                    status=response.status,
                                    error=text[:500]
                                )
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(
                            "ai_log_processing_error",
                            index=i,
                            order_id=ai_log.get("order_id", "UNKNOWN"),
                            error=str(e)
                        )
                        continue
                
                logger.info("ai_log_upload_complete")
                
            except Exception as e:
                logger.error("upload_failed", error=str(e))


async def main():
    """Main entry point"""
    print("Starting AI log upload to WEEX...")
    print(f"Using API key: {settings.WEEX_API_KEY[:10]}..." if settings.WEEX_API_KEY else "No API key configured")
    print(f"Dry run: False (set dry_run=True in WeexClient for testing)")
    print()
    
    await upload_ai_logs()
    
    print("\nUpload complete!")


if __name__ == "__main__":
    asyncio.run(main())
