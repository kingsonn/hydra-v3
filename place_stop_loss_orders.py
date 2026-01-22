#!/usr/bin/env python3
"""
Script to place stop loss orders for short positions using Weex API
"""

import time
import hmac
import hashlib
import base64
import requests
import json
import os
import uuid
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.connection import create_connection

load_dotenv()

# API Credentials
api_key = os.getenv("WEEX_API_KEY")
secret_key = os.getenv("WEEX_SECRET_KEY")
access_passphrase = os.getenv("WEEX_PASSPHRASE")

# Configuration
PROXY = os.getenv("WEEX_PROXY", None)
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
FORCE_IPV4 = os.getenv("WEEX_FORCE_IPV4", "true").lower() == "true"
BASE_URL = "https://api-contract.weex.com"

# Force IPv4 connection
def create_ipv4_connection(address, timeout, source_address):
    import socket
    host, port = address
    err = None
    for res in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            return sock
        except socket.error as _:
            err = _
            if sock is not None:
                sock.close()
    if err is not None:
        raise err
    else:
        raise socket.error("getaddrinfo returns an empty list")

class IPv4Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['socket_options'] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        return super().init_poolmanager(*args, **kwargs)

def get_session():
    session = requests.Session()
    if FORCE_IPV4:
        import socket as socket_module
        original_create_connection = create_connection
        def patched_create_connection(address, *args, **kwargs):
            host, port = address
            return create_ipv4_connection(address, kwargs.get('timeout', socket_module._GLOBAL_DEFAULT_TIMEOUT), kwargs.get('source_address'))
        session.mount('http://', HTTPAdapter())
        session.mount('https://', HTTPAdapter())
    return session

SESSION = get_session()

def generate_signature(secret_key, timestamp, method, request_path, query_string, body):
    message = timestamp + method.upper() + request_path + query_string + str(body)
    signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()

def send_request_post(api_key, secret_key, access_passphrase, method, request_path, query_string, body):
    timestamp = str(int(time.time() * 1000))
    body = json.dumps(body)
    signature = generate_signature(secret_key, timestamp, method, request_path, query_string, body)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    url = BASE_URL + request_path
    response = SESSION.post(url, headers=headers, data=body, timeout=30, proxies=PROXIES)
    return response

def upload_ai_log(api_key, secret_key, access_passphrase, order_id):
    """Upload AI log for the stop loss order"""
    
    ai_log_data = {
        "orderId": order_id,
        "stage": "Risk Management",
        "model": "Rules Based",
        "input": {"user_action": "FULL_RISK_MODE", "reason": "User changed risk mode to FULL_RISK on the dashboard"},
        "output": {"status": "completed", "risk_level": "FULL_RISK"},
        "explanation": "User changed risk mode to FULL_RISK on the dashboard"
    }
    
    request_path = "/capi/v2/ai/upload"
    query_string = ""
    
    try:
        response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, ai_log_data)
        print(f"AI Log Upload Status: {response.status_code}")
        if response.status_code == 200:
            print(f"AI Log uploaded successfully: {response.text}")
        else:
            print(f"AI Log upload failed: {response.text}")
    except Exception as e:
        print(f"Error uploading AI log: {e}")

def place_stop_loss_orders():
    """Place stop loss orders for specified symbols"""
    
    # Stop loss configurations
    stop_configs = [
        {"symbol": "cmt_btcusdt", "trigger_price": "130000", "size": "0.019"},
        {"symbol": "cmt_solusdt", "trigger_price": "200", "size": "13.5"},
        {"symbol": "cmt_ethusdt", "trigger_price": "4000", "size": "0.586"},
        {"symbol": "cmt_adausdt", "trigger_price": "0.60", "size": "3121"}
    ]
    
    request_path = "/capi/v2/order/placeTpSlOrder"
    query_string = ""
    
    for config in stop_configs:
        print(f"\n{'='*60}")
        print(f"Placing stop loss order for {config['symbol']}")
        print(f"{'='*60}")
        
        # Generate unique client order ID
        client_order_id = f"sl_{int(time.time() * 1000)}_{config['symbol'].replace('cmt_', '')}"
        
        # Build request body
        body = {
            "symbol": config["symbol"],
            "clientOrderId": client_order_id,
            "planType": "loss_plan",  # Stop-loss plan order
            "triggerPrice": config["trigger_price"],
            "executePrice": "0",  # Market price on trigger
            "size": config["size"],
            "positionSide": "short",  # Short position
            "marginMode": 1  # Cross Mode
        }
        
        print(f"Order Details: {json.dumps(body, indent=2)}")
        
        try:
            response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Text: {response.text}")
            
            if response.status_code == 200:
                response_json = response.json()
                print(f"✅ Stop loss order placed successfully!")
                print(f"Response JSON:\n{json.dumps(response_json, indent=2)}")
                
                # Extract order ID for AI log
                if response_json.get("code") == "00000" and response_json.get("data"):
                    order_id = str(response_json["data"].get("orderId", client_order_id))
                    print(f"Order ID: {order_id}")
                    
                    # Upload AI log
                    upload_ai_log(api_key, secret_key, access_passphrase, order_id)
                else:
                    print(f"⚠️  Order may have failed: {response_json}")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                if response.status_code == 521:
                    print("⚠️  Status 521 usually means:")
                    print("   - Server is down/unreachable")
                    print("   - IP address is blocked/restricted")
                    print("   - Cloudflare protection is blocking the request")
                    print("   - VPN/Proxy may be required")
                
        except Exception as e:
            print(f"❌ Error placing stop loss order: {e}")
        
        # Small delay between orders
        time.sleep(1)

def main():
    print("="*60)
    print("WEEX STOP LOSS ORDER PLACEMENT SCRIPT")
    print("="*60)
    print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT SET")
    print(f"Secret Key: {secret_key[:20]}..." if secret_key else "Secret Key: NOT SET")
    print(f"Passphrase: {access_passphrase[:10]}..." if access_passphrase else "Passphrase: NOT SET")
    print(f"Proxy: {PROXY if PROXY else 'Not configured'}")
    print(f"Force IPv4: {'Yes' if FORCE_IPV4 else 'No'}")
    
    if not all([api_key, secret_key, access_passphrase]):
        print("\n❌ ERROR: Missing API credentials in .env file")
        print("Please ensure WEEX_API_KEY, WEEX_SECRET_KEY, and WEEX_PASSPHRASE are set")
        return
    
    print(f"\n🎯 Placing stop loss orders for SHORT positions:")
    print("   - BTCUSDT: Stop loss at $130,000")
    print("   - SOLUSDT: Stop loss at $200")
    print("   - ETHUSDT: Stop loss at $4,000")
    print("   - ADAUSDT: Stop loss at $0.60")
    
    # Confirm before proceeding
    confirm = input("\n⚠️  This will place real stop loss orders. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Operation cancelled by user")
        return
    
    place_stop_loss_orders()
    
    print("\n" + "="*60)
    print("STOP LOSS ORDER PLACEMENT COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
