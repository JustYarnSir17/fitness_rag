# app/tools/web_tools.py
from langchain_core.tools import tool
import os, json, requests, time

PROVIDER = os.getenv("WEB_SEARCH_PROVIDER","tavily").lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

@tool("web_search", return_direct=False)
def web_search(query_json: str) -> str:
    """웹 검색 제공자(Tavily/SerpAPI)를 사용해 쿼리 결과를 JSON으로 반환합니다.
    입력: {"query":"...", "max_results":5}
    반환: {"used": bool, "provider": "...", "elapsed_ms": int, "results":[{title,url,snippet}...], "error": "...?(optional)"}
    """
    t0 = time.perf_counter()
    args = json.loads(query_json)
    q = args.get("query",""); n = int(args.get("max_results",5))
    used = False; results = []; error = None; provider = PROVIDER

    try:
        if provider == "tavily" and TAVILY_API_KEY:
            used = True
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": q, "max_results": n}
            )
            r.raise_for_status()
            for it in r.json().get("results",[]):
                results.append({"title": it.get("title"), "url": it.get("url"), "snippet": it.get("content")})
        elif provider == "serpapi" and SERPAPI_API_KEY:
            used = True
            r = requests.get(
                "https://serpapi.com/search.json",
                params={"q": q, "api_key": SERPAPI_API_KEY, "num": n}
            )
            r.raise_for_status()
            for it in r.json().get("organic_results",[])[:n]:
                results.append({"title": it.get("title"), "url": it.get("link"), "snippet": it.get("snippet")})
        else:
            error = "No provider/key configured"
    except Exception as e:
        error = str(e)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return json.dumps({
        "used": used, "provider": provider, "elapsed_ms": elapsed_ms,
        "results": results, "count": len(results), **({"error": error} if error else {})
    }, ensure_ascii=False)

@tool("corroborate_answer", return_direct=False)
def corroborate_answer(input_json: str) -> str:
    """질문과 초안 답변을 받아 웹 검색으로 증거를 수집해 함께 반환합니다.
    입력: {"question":"...", "draft":"...", "max_results":3}
    반환: {"question":..., "draft":..., "evidence":[...], "meta":{"used":bool,"provider":"...","elapsed_ms":int,"error":"...?","count":int}}
    """
    t0 = time.perf_counter()
    args = json.loads(input_json)
    q = args.get("question",""); draft = args.get("draft",""); n = int(args.get("max_results",3))
    web_raw = web_search(json.dumps({"query": q, "max_results": n}))
    web = json.loads(web_raw)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return json.dumps({
        "question": q,
        "draft": draft,
        "evidence": web.get("results", []),
        "meta": {
            "used": bool(web.get("used")),
            "provider": web.get("provider"),
            "elapsed_ms": web.get("elapsed_ms", 0),
            "error": web.get("error"),
            "count": web.get("count", 0),
            "wrapper_elapsed_ms": elapsed_ms
        }
    }, ensure_ascii=False)
