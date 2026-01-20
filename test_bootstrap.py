"""Quick test for bootstrap SSL connection"""
import asyncio
from src.collectors.bootstrap import AlphaDataBootstrap

async def test():
    b = AlphaDataBootstrap(['BTCUSDT'])
    result = await b.bootstrap_all()
    print(f'Bootstrap result: {result}')
    print(f'OI samples: {len(b.oi_data["BTCUSDT"].history)}')
    print(f'Funding samples: {len(b.funding_data["BTCUSDT"].history)}')
    print(f'Price bars: {len(b.price_data["BTCUSDT"].bars_1h)}')

if __name__ == "__main__":
    asyncio.run(test())
