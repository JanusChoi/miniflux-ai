import json
import time
from openai import OpenAI

from common import logger
from common.config import Config
from core.get_ai_result import get_ai_result

config = Config()
llm_client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)

def generate_daily_news():
    logger.info('Generating daily news')
    # fetch entries.json
    try:
        with open('entries.json', 'r') as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    contents = '\n'.join([i['content'] for i in entries])
    # greeting
    greeting = get_ai_result(config.ai_news_prompts['greeting'], time.strftime('%B %d, %Y at %I:%M %p'))
    # summary_block
    summary_block = get_ai_result(config.ai_news_prompts['summary_block'], contents)
    # summary
    summary = get_ai_result(config.ai_news_prompts['summary'], summary_block)

    response_content = greeting + '\n\n### 🌐Summery\n' + summary + '\n\n### 📝News\n' + summary_block

    logger.info('Generated daily news: ' + response_content)

    # empty entries.json
    with open('entries.json', 'w') as f:
        json.dump([], f, indent=4, ensure_ascii=False)

    with open('ai_news.json', 'w') as f:
        json.dump(response_content, f, indent=4, ensure_ascii=False)

    # Todo trigger miniflux feed update
