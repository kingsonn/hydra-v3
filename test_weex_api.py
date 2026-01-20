import time
import hmac
import hashlib
import base64
import requests
import json
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.connection import create_connection

load_dotenv()

api_key = os.getenv("WEEX_API_KEY")
secret_key = os.getenv("WEEX_SECRET_KEY")
access_passphrase = os.getenv("WEEX_PASSPHRASE")

# Optional: Set proxy if needed (format: http://ip:port or socks5://ip:port)
PROXY = os.getenv("WEEX_PROXY", None)
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None

# Force IPv4 (set to True if Weex whitelisted your IPv4 address)
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


# Configure session with IPv4 forcing if enabled
def get_session():
    session = requests.Session()
    if FORCE_IPV4:
        import socket as socket_module
        # Monkey patch to force IPv4
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


def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
    message = timestamp + method.upper() + request_path + query_string
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


def send_request_get(api_key, secret_key, access_passphrase, method, request_path, query_string):
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature_get(secret_key, timestamp, method, request_path, query_string)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    url = BASE_URL + request_path + query_string
    response = SESSION.get(url, headers=headers, timeout=30, proxies=PROXIES)
    return response


def test_connectivity():
    print("\n" + "="*60)
    print("TEST 0: API Connectivity Check")
    print("="*60)
    
    try:
        response = SESSION.get(BASE_URL, timeout=10, proxies=PROXIES)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        if PROXY:
            print(f"Using Proxy: {PROXY}")
        if FORCE_IPV4:
            print(f"Forcing IPv4: ✅")
        print(f"Server is reachable: ✅")
        return True
    except requests.exceptions.Timeout:
        print("❌ Connection timeout - server may be down or unreachable")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("⚠️  This could indicate:")
        print("   - IP address is blocked")
        print("   - Firewall/network restrictions")
        print("   - VPN required")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_get_account_assets():
    print("\n" + "="*60)
    print("TEST 1: GET /capi/v2/account/position/singlePosition")
    print("="*60)
    
    request_path = "/capi/v2/account/position/singlePosition"
    query_string = "?symbol=cmt_btcusdt"
    
    try:
        response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Response JSON:\n{json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.status_code == 521:
                print("⚠️  Status 521 usually means:")
                print("   - Server is down/unreachable")
                print("   - IP address is blocked/restricted")
                print("   - Cloudflare protection is blocking the request")
                print("   - VPN/Proxy may be required")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response Text: {response.text if 'response' in locals() else 'N/A'}")
        return False


def test_get_order_detail():
    print("\n" + "="*60)
    print("TEST 3: GET /capi/v2/order/detail")
    print("="*60)
    
    request_path = "/capi/v2/order/detail"
    query_string = "?orderId=702947048126677006"
    
    try:
        response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Response JSON:\n{json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.status_code == 521:
                print("⚠️  Status 521 usually means:")
                print("   - Server is down/unreachable")
                print("   - IP address is blocked/restricted")
                print("   - Cloudflare protection is blocking the request")
                print("   - VPN/Proxy may be required")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response Text: {response.text if 'response' in locals() else 'N/A'}")
        return False


def test_get_all_positions():
    print("\n" + "="*60)
    print("TEST 4: GET /capi/v2/account/position/allPosition")
    print("="*60)
    
    request_path = "/capi/v2/account/position/allPosition"
    query_string = ""
    
    try:
        response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Response JSON:\n{json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.status_code == 521:
                print("⚠️  Status 521 usually means:")
                print("   - Server is down/unreachable")
                print("   - IP address is blocked/restricted")
                print("   - Cloudflare protection is blocking the request")
                print("   - VPN/Proxy may be required")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response Text: {response.text if 'response' in locals() else 'N/A'}")
        return False


def test_place_order():
    print("\n" + "="*60)
    print("TEST 2: POST /capi/v2/order/placeOrder")
    print("="*60)
    
    request_path = "/capi/v2/order/placeOrder"
    query_string = ""
    
    # 10 USDT position size
    # For BTC at ~$95000, size = 10 / 95000 ≈ 0.0001 BTC
    # Using match_price=1 to execute at market price
    body = {
        "symbol": "cmt_bnbusdt",
        "client_oid": str(int(time.time() * 1000000)),
        "size": "2",  # ~10 USDT at current BTC price
        "type": "2",  # 1=open long, 2=open short, 3=close long, 4=close short
        "order_type": "1",  # 0=limit order, 1=market order
        "match_price": "1",  # 1=use market price, 0=use specified price
        "price": ""  # Empty when using match_price=1
    }
    
    print(f"Order Details: {json.dumps(body, indent=2)}")
    
    try:
        response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Response JSON:\n{json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            if response.status_code == 521:
                print("⚠️  Status 521 usually means:")
                print("   - Server is down/unreachable")
                print("   - IP address is blocked/restricted")
                print("   - Cloudflare protection is blocking the request")
                print("   - VPN/Proxy may be required")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response Text: {response.text if 'response' in locals() else 'N/A'}")
        return False


def main():
    print("="*60)
    print("WEEX API TEST SCRIPT")
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
    
    connectivity_ok = test_connectivity()
    
    if not connectivity_ok:
        print("\n⚠️  Skipping authenticated tests due to connectivity issues")
        print("💡 Try using a VPN (Hong Kong/Singapore region recommended)")
        return
    
    # test1_passed = test_get_account_assets()
    
    # test2_passed = test_place_order()
    
    # test3_passed = test_get_order_detail()
    
    test4_passed = test_get_all_positions()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    # print(f"✅ GET /capi/v2/account/position/singlePosition: {'PASSED' if test1_passed else '❌ FAILED'}")
    # print(f"✅ POST /capi/v2/order/placeOrder: {'PASSED' if test2_passed else '❌ FAILED'}")
    # print(f"✅ GET /capi/v2/order/detail: {'PASSED' if test3_passed else '❌ FAILED'}")
    print(f"✅ GET /capi/v2/account/position/allPosition: {'PASSED' if test4_passed else '❌ FAILED'}")
    print("="*60)


if __name__ == '__main__':
    main()
