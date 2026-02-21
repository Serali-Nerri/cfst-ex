import asyncio
import httpx
import json

API_URL = "http://127.0.0.1:8317/v1/chat/completions"
API_KEY = "sk-Xqt9KSE1H6nrLRZOCalPnilk3WZT1vCsYNkoLCLv9QUHE"

async def test_api(payload):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(API_URL, headers=headers, json=payload, timeout=60.0)
            print(f"Status Code: {resp.status_code}")
            try:
                data = resp.json()
                print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
            except:
                print(resp.text)
        except Exception as e:
            print(f"Error: {e}")

async def main():
    print("Test 1: Normal request")
    await test_api({
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })

    print("\nTest 2: With xhigh parameter")
    await test_api({
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "xhigh": True
    })

    print("\nTest 3: With reasoning_effort high")
    await test_api({
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "reasoning_effort": "high"
    })

if __name__ == "__main__":
    asyncio.run(main())
