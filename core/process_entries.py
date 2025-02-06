import json
import markdown
from markdownify import markdownify as md
from openai import OpenAI
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any
import traceback
import os
from datetime import datetime

from common.config import Config
from common.logger import logger
from core.entry_filter import filter_entry

config = Config()
llm_client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
llm_r1_client = OpenAI(base_url=config.llm_r1_base_url, api_key=config.llm_r1_api_key)
file_lock = threading.Lock()

@dataclass
class ProcessResult:
    title: Optional[str] = None
    content: Optional[str] = None
    type: str = ""
    error: Optional[str] = None

class EntryProcessor:
    def __init__(self, config, llm_client, llm_r1_client):
        self.config = config
        self.llm_client = llm_client
        self.llm_r1_client = llm_r1_client
        self.llm_executor = ThreadPoolExecutor(max_workers=config.llm_max_workers)
        self.llm_r1_executor = ThreadPoolExecutor(max_workers=config.llm_r1_max_workers)

    def translate_title(self, title: str) -> ProcessResult:
        """同步处理标题翻译"""
        try:
            logger.info(f"调用LLM翻译标题: {title[:50]}...")
            messages = [
                {"role": "system", "content": self.config.agents['translate_title']['prompt']},
                {"role": "user", "content": title}
            ]
            
            # 记录调用参数
            logger.debug(f"LLM调用参数: base_url={self.llm_client.base_url}, model={self.config.llm_model}, timeout={self.config.llm_timeout}")
            
            try:
                completion = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    timeout=self.config.llm_timeout
                )
            except Exception as e:
                logger.error(f"LLM API调用失败: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                raise
            
            translated_title = completion.choices[0].message.content
            logger.info(f"标题翻译结果: {translated_title}")
            return ProcessResult(
                title=translated_title,
                type="translate_title"
            )
        except Exception as e:
            logger.error(f"标题翻译出错: {str(e)}")
            return ProcessResult(error=str(e), type="translate_title")

    def translate_content(self, content: str) -> ProcessResult:
        """同步处理全文翻译"""
        try:
            logger.info("开始调用LLM翻译全文...")
            messages = [
                {"role": "system", "content": self.config.agents['translate']['prompt']},
                {"role": "user", "content": md(content)}
            ]
            
            completion = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                timeout=self.config.llm_timeout
            )
            
            logger.info("全文翻译完成")
            return ProcessResult(
                content=completion.choices[0].message.content,
                type="translate"
            )
        except Exception as e:
            logger.error(f"全文翻译出错: {str(e)}")
            return ProcessResult(error=str(e), type="translate")

    def generate_summary(self, content: str) -> ProcessResult:
        """同步处理摘要生成"""
        try:
            logger.info("开始调用LLM生成摘要...")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.config.agents['summary']['prompt'].replace('${content}', content)}
            ]
            
            completion = self.llm_r1_client.chat.completions.create(
                model=self.config.llm_r1_model,
                messages=messages,
                timeout=self.config.llm_r1_timeout
            )
            
            think_content, summary_content = split_content(completion.choices[0].message.content)
            logger.info("摘要生成完成")
            return ProcessResult(
                content={'think': think_content, 'summary': summary_content},
                type="summary"
            )
        except Exception as e:
            logger.error(f"摘要生成出错: {str(e)}")
            return ProcessResult(error=str(e), type="summary")

    def build_final_content(self, original_content: str, results: Dict[str, ProcessResult]) -> str:
        llm_result = ""
        
        # 处理翻译结果
        if 'translate' in results and not results['translate'].error:
            llm_result += (
                f"{self.config.agents['translate']['title']}：\n" +
                markdown.markdown(results['translate'].content) +
                '<hr><br />'
            )

        # 处理摘要结果
        if 'summary' in results and not results['summary'].error:
            summary_data = results['summary'].content
            if self.config.agents['summary']['style_block']:
                if summary_data['think']:
                    llm_result += (
                        '<details><summary>🤔 AI思考过程</summary><pre style="white-space: pre-wrap;"><code>\n'
                        + summary_data['think']
                        + '\n</code></pre></details><br />'
                        + self.config.agents['summary']['title'] + '：\n'
                        + markdown.markdown(summary_data['summary'], extensions=['extra'])
                        + '<hr><br />'
                    )
                else:
                    llm_result += (
                        self.config.agents['summary']['title'] + '：\n'
                        + markdown.markdown(summary_data['summary'])
                        + '<hr><br />'
                    )

        return mark_as_ai_processed(llm_result + original_content)

def has_ai_processed(content):
    """检查内容是否已经被AI处理过"""
    if not content:
        return False
    return content.startswith('<!-- AI_PROCESSED -->')

def mark_as_ai_processed(content):
    """标记内容为已处理"""
    return f'<!-- AI_PROCESSED -->\n{content}'

def split_content(response_content):
    """分离AI思考和摘要内容"""
    logger.debug(f"Original response content: {response_content[:200]}...")
    
    # 1. 先提取思考内容
    think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    # 2. 获取摘要内容（移除整个 think 标签块）
    summary_content = re.sub(r'<think>.*?</think>', '', response_content, re.DOTALL).strip()
    
    # 3. 移除摘要内容中可能存在的 <think> 标签
    if summary_content.startswith('<think>'):
        summary_content = re.sub(r'^<think>.*?</think>\s*', '', summary_content, flags=re.DOTALL)
    
    # 4. 清理内容，确保正确的 Markdown 格式
    def clean_content(content):
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # 保留列表项的前导空格
            if line.lstrip().startswith(('1.', '2.', '3.', '-', '*')):
                # 确保列表项后有一个空行
                cleaned_lines.append(line)
                cleaned_lines.append('')
            else:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    think_content = clean_content(think_content)
    summary_content = clean_content(summary_content)
    
    logger.debug(f"Cleaned summary content:\n{summary_content}")
    return think_content, summary_content

def get_entry_status(entry_id: int) -> Optional[Dict]:
    """获取文章的处理状态
    
    Args:
        entry_id: 文章ID
    
    Returns:
        包含处理状态的字典，如果文章不存在则返回 None
    """
    try:
        entries_file = 'entries.json'
        if not os.path.exists(entries_file):
            return None
            
        with file_lock:
            with open(entries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    if entry.get('id') == entry_id:
                        return entry
        return None
        
    except Exception as e:
        logger.error(f"获取文章状态时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_entry(miniflux_client, entry: Dict[str, Any]) -> None:
    """处理单个文章条目"""
    logger.info(f"开始处理文章: {entry['title']}")
    
    # 检查是否已经完全处理过
    entry_status = get_entry_status(entry['id'])
    if entry_status and entry_status.get('processed', {}).get('title_translated') and \
       entry_status.get('processed', {}).get('content_translated') and \
       entry_status.get('processed', {}).get('summary_generated'):
        logger.info(f"Entry {entry['title']} 已完全处理，跳过")
        return

    processor = EntryProcessor(config, llm_client, llm_r1_client)
    results = {}
    translated_content = None

    try:
        # 1. 首先处理标题翻译
        if 'translate_title' in config.agents and \
           not (entry_status and entry_status.get('processed', {}).get('title_translated')):
            logger.info(f"开始翻译标题: {entry['title']}")
            with processor.llm_executor as executor:
                future = executor.submit(
                    processor.translate_title,
                    entry['title']
                )
                result = future.result(timeout=config.llm_timeout)
                if not result.error:
                    translated_title = result.title
                    entry['title'] = f"{entry['title']} | {translated_title}"
                    miniflux_client.update_entry(entry['id'], title=entry['title'])
                    save_summary_to_json(entry, translated_title=translated_title)
                    logger.info(f"标题翻译完成: {entry['title']}")
                else:
                    logger.error(f"标题翻译失败: {result.error}")
                results['translate_title'] = result

        # 2. 处理全文翻译
        if 'translate' in config.agents and \
           not (entry_status and entry_status.get('processed', {}).get('content_translated')) and \
           filter_entry(config, ('translate', config.agents['translate']), entry):
            logger.info(f"开始翻译全文: {entry['title']}")
            with processor.llm_executor as executor:
                future = executor.submit(
                    processor.translate_content,
                    entry['content']
                )
                result = future.result(timeout=config.llm_timeout)
                if not result.error:
                    translated_content = result.content
                    entry['content'] = translated_content
                    miniflux_client.update_entry(entry['id'], content=translated_content)
                    save_summary_to_json(entry, translated_content=translated_content)
                    logger.info(f"全文翻译完成: {entry['title']}")
                else:
                    logger.error(f"全文翻译失败: {result.error}")
                results['translate'] = result

        # 3. 处理摘要
        if 'summary' in config.agents and \
           not (entry_status and entry_status.get('processed', {}).get('summary_generated')) and \
           filter_entry(config, ('summary', config.agents['summary']), entry):
            logger.info(f"开始生成摘要: {entry['title']}")
            content_for_summary = translated_content if translated_content else entry['content']
            with processor.llm_r1_executor as executor:
                future = executor.submit(
                    processor.generate_summary,
                    content_for_summary
                )
                result = future.result(timeout=config.llm_r1_timeout)
                if not result.error:
                    summary_data = result.content
                    save_summary_to_json(entry, summary=summary_data)
                    logger.info(f"摘要生成完成: {entry['title']}")
                else:
                    logger.error(f"摘要生成失败: {result.error}")
                results['summary'] = result

        # 构建最终内容
        if any(not r.error for r in results.values()):
            logger.info(f"开始构建最终内容: {entry['title']}")
            final_content = processor.build_final_content(entry['content'], results)
            miniflux_client.update_entry(entry['id'], content=final_content)
            logger.info(f"内容更新完成: {entry['title']}")

        logger.info(f"文章处理完成: {entry['title']}")

    except Exception as e:
        logger.error(f"Error processing entry {entry['title']}: {e}")
        logger.error(traceback.format_exc())

def save_summary_to_json(entry: Dict[str, Any], summary: str = None, translated_title: str = None, translated_content: str = None) -> None:
    """保存文章信息到 entries.json 文件，包含处理状态
    
    Args:
        entry: Miniflux 的文章条目数据
        summary: 生成的摘要内容
        translated_title: 翻译后的标题
        translated_content: 翻译后的全文内容
    """
    try:
        entries_file = 'entries.json'
        
        entry_data = {
            'id': entry.get('id', 0),
            'url': entry.get('url', ''),
            'published_at': entry.get('published_at', ''),
            'feed': {
                'id': entry.get('feed', {}).get('id'),
                'title': entry.get('feed', {}).get('title', ''),
                'category': entry.get('feed', {}).get('category', {}).get('title', '')
            },
            'original': {
                'title': entry.get('title', ''),
                'content': entry.get('content', ''),
                'author': entry.get('author', ''),
                'tags': entry.get('tags', [])
            },
            'processed': {
                'title_translated': bool(translated_title),
                'translated_title': translated_title,
                'content_translated': bool(translated_content),
                'translated_content': translated_content,
                'summary_generated': bool(summary),
                'summary': summary,
                'last_updated': datetime.utcnow().isoformat()
            }
        }

        with file_lock:
            try:
                # 读取现有数据
                data = []
                if os.path.exists(entries_file):
                    with open(entries_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                # 更新或添加条目
                exists = False
                for i, item in enumerate(data):
                    if item.get('id') == entry_data['id']:
                        # 合并现有处理状态
                        if not entry_data['processed']['title_translated']:
                            entry_data['processed']['title_translated'] = item['processed'].get('title_translated', False)
                            entry_data['processed']['translated_title'] = item['processed'].get('translated_title')
                        if not entry_data['processed']['content_translated']:
                            entry_data['processed']['content_translated'] = item['processed'].get('content_translated', False)
                            entry_data['processed']['translated_content'] = item['processed'].get('translated_content')
                        if not entry_data['processed']['summary_generated']:
                            entry_data['processed']['summary_generated'] = item['processed'].get('summary_generated', False)
                            entry_data['processed']['summary'] = item['processed'].get('summary')
                        
                        data[i] = entry_data
                        exists = True
                        break

                if not exists:
                    data.append(entry_data)

                # 写入文件
                with open(entries_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                logger.debug(f"Entry {entry_data['id']} 已更新到 {entries_file}")

            except json.JSONDecodeError:
                logger.error(f"JSON 解析错误: {entries_file}")
                with open(entries_file, 'w', encoding='utf-8') as f:
                    json.dump([entry_data], f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"保存条目时出错: {str(e)}")
        logger.error(traceback.format_exc())
