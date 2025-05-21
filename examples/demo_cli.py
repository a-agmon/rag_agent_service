"""Tiny CLI to chat with SelfRAG."""
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
import asyncio

from selfrag import SelfRAG

rag = SelfRAG()

logging.info('SelfRAG CLI started.')

async def chat():
    while True:
        try:
            q = input("❓ ")
            logging.info(f'Received question: {q}')
        except (EOFError, KeyboardInterrupt):
            logging.info('Exiting SelfRAG CLI.')
            break
        if not q.strip():
            continue
        try:
            result = await rag._graph.ainvoke({"query": q})
            answer = result.get("answer")
            print("🟢", answer)
            logging.info('Answer returned.')
        except Exception as e:
            logging.error(f'Error during answer generation: {e}')

if __name__ == "__main__":
    asyncio.run(chat()) 