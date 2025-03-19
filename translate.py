#!/usr/bin/env python3
import sys
import re
import os
import subprocess
import time
import argparse
from typing import List, Dict, Any
import asyncio
from tqdm.asyncio import tqdm_asyncio

# LangChain imports
from langchain_ollama import ChatOllama  # Updated import
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage, HumanMessage
from pydantic import BaseModel, Field


class SubtitleEntry(BaseModel):
    """Model for a subtitle entry."""
    index: int = Field(description="The subtitle index number")
    text: str = Field(description="The translated subtitle text")

class SubtitleBatch(BaseModel):
    """Model for a batch of translated subtitles."""
    entries: List[SubtitleEntry] = Field(description="List of translated subtitle entries")

def parse_srt(file_path):
    """Parse an SRT subtitle file into a list of subtitle entries."""
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
    
    # Split by double newline to separate subtitle blocks
    subtitle_blocks = re.split(r'\n\n+', content.strip())
    subtitles = []
    
    for block in subtitle_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                timing = lines[1]
                text = '\n'.join(lines[2:])
                subtitles.append({
                    'index': index, 
                    'timing': timing, 
                    'text': text
                })
            except ValueError:
                # Skip invalid blocks
                continue
    
    return subtitles

def batch_subtitles(subtitles, batch_size=5):
    """Group subtitles into batches for more efficient translation."""
    batches = []
    current_batch = []
    current_batch_text_length = 0
    max_chars_per_batch = 500  # Limit characters per batch
    
    for sub in subtitles:
        # If adding this subtitle would exceed character limit or batch size
        if (current_batch_text_length + len(sub['text']) > max_chars_per_batch 
                or len(current_batch) >= batch_size) and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_text_length = 0
        
        current_batch.append(sub)
        current_batch_text_length += len(sub['text'])
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def write_srt(subtitles, output_path):
    """Write subtitles back to SRT format."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for i, sub in enumerate(subtitles):
            if i > 0:
                file.write('\n\n')
            file.write(f"{sub['index']}\n")
            file.write(f"{sub['timing']}\n")
            file.write(sub['text'])

async def translate_batch(batch, llm, source_lang, target_lang):
    """Translate a batch of subtitles using LangChain."""
    # Create input for the model
    subtitle_texts = []
    for sub in batch:
        subtitle_texts.append(f"[{sub['index']}] {sub['text']}")
    
    combined_text = "\n\n".join(subtitle_texts)
    
    # Create translation prompt
    source_lang_prompt = f"from {source_lang}" if source_lang.lower() != "auto" else ""
    template = f"""Translate the following subtitles {source_lang_prompt} to {target_lang}.
Preserve the original meaning, formatting, and line breaks as accurately as possible.
Always keep the subtitle index numbers in square brackets.

SUBTITLES:
{{text}}

TRANSLATION:"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", template)
    ])
    
    # Create the chain
    chain = prompt | llm
    
    # Execute the chain
    result = await chain.ainvoke({"text": combined_text})
    
    # Parse the results
    translated_text = result.content if hasattr(result, 'content') else str(result)
    
    # Process the translated text
    pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)'
    matches = re.finditer(pattern, translated_text, re.DOTALL)
    
    # Update the batch with translations
    processed = set()
    for match in matches:
        sub_index = int(match.group(1))
        sub_text = match.group(2).strip()
        
        # Find the corresponding subtitle in the batch
        for sub in batch:
            if sub['index'] == sub_index and sub_index not in processed:
                sub['text'] = sub_text
                processed.add(sub_index)
                break
    
    # Return the updated batch
    return batch

async def translate_individual(subtitle, llm, source_lang, target_lang):
    """Translate a single subtitle using LangChain."""
    # Create translation prompt
    source_lang_prompt = f"from {source_lang}" if source_lang.lower() != "auto" else ""
    template = f"""Translate the following subtitle {source_lang_prompt} to {target_lang}.
Preserve the original meaning, formatting, and line breaks as accurately as possible.

SUBTITLE:
{{text}}

TRANSLATION:"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", template)
    ])
    
    # Create the chain
    chain = prompt | llm
    
    # Execute the chain
    result = await chain.ainvoke({"text": subtitle['text']})
    
    # Get the translation
    translated_text = result.content if hasattr(result, 'content') else str(result)
    
    # Update the subtitle
    subtitle['text'] = translated_text.strip()
    
    return subtitle

async def process_subtitles(subtitles, batch_size, individual, source_lang, target_lang, model_name, temperature):
    """Process all subtitles with appropriate method."""
    # Initialize the LLM
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url="http://localhost:11434",
    )
    
    try:
        if individual:
            # Process subtitles individually
            tasks = []
            for sub in subtitles:
                # Add a small delay to avoid overwhelming Ollama
                await asyncio.sleep(0.05)
                tasks.append(translate_individual(sub, llm, source_lang, target_lang))
            
            # Process with progress bar
            return await tqdm_asyncio.gather(*tasks, desc="Translating subtitles")
        else:
            # Process in batches
            batches = batch_subtitles(subtitles, batch_size)
            print(f"Processing in {len(batches)} batches")
            
            results = []
            for batch in tqdm_asyncio(batches, desc="Translating batches"):  # Corrected method
                try:
                    translated_batch = await translate_batch(batch, llm, source_lang, target_lang)
                    results.extend(translated_batch)
                    # Add a small delay between batches
                    await asyncio.sleep(0.2)
                except Exception as e:
                    print(f"\nError in batch translation: {e}")
                    print("Falling back to individual translation for this batch")
                    
                    # Fall back to individual translation
                    for sub in batch:
                        try:
                            translated_sub = await translate_individual(sub, llm, source_lang, target_lang)
                            results.append(translated_sub)
                            await asyncio.sleep(0.05)
                        except Exception as sub_e:
                            print(f"Error translating subtitle {sub['index']}: {sub_e}")
                            results.append(sub)  # Keep original if translation fails
            
            return results
    finally:
        # Unload the model
        subprocess.run(["ollama", "stop", model_name])

# Check if ollama serve is running
def ensure_ollama_serve():
    try:
        # Check if 'ollama serve' is running
        result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 'ollama serve' is not running.")
            print("➡️  Please start it manually by running: `ollama serve`")
            sys.exit(1)
        else:
            print("✅ 'ollama serve' is already running.")
    except Exception as e:
        print(f"⚠️ Error checking 'ollama serve': {e}")
        sys.exit(1)


async def main_async():
    parser = argparse.ArgumentParser(description='Translate SRT subtitle files using LangChain and Ollama')
    parser.add_argument('file', help='SRT file to translate')
    parser.add_argument('-i', '--input-language', default='Auto', 
                        help='Source language (default: Auto-detect)')
    parser.add_argument('-o', '--output-language', default='English', 
                        help='Target language (default: English)')
    parser.add_argument('-m', '--model', default='vanilj/gemma-2-ataraxy-9b:IQ2_M', 
                        help='Ollama model to use (default: deepseekr1:14b)')
    parser.add_argument('-f', '--output-file', 
                        help='Output file name (default: input_file_translated.srt)')
    parser.add_argument('-b', '--batch-size', type=int, default=5,
                        help='Number of subtitles to translate at once (default: 5)')
    parser.add_argument('-t', '--temperature', type=float, default=0.3,
                        help='Model temperature for translation (default: 0.3)')
    parser.add_argument('--individual', action='store_true',
                        help='Translate each subtitle individually (slower but potentially more accurate)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Set default output filename if not specified
    if not args.output_file:
        base_name, ext = os.path.splitext(args.file)
        args.output_file = f"{base_name}_{args.output_language.lower()}{ext}"
    
    ensure_ollama_serve()

    print(f"Parsing subtitle file: {args.file}")
    subtitles = parse_srt(args.file)
    print(f"Found {len(subtitles)} subtitle entries")
    
    print(f"Translating from {args.input_language} to {args.output_language} using {args.model}...")
    
    # Process all subtitles
    try:
        translated_subtitles = await process_subtitles(
            subtitles, 
            args.batch_size, 
            args.individual, 
            args.input_language, 
            args.output_language, 
            args.model,
            args.temperature
        )
        
        # Write the translated subtitles to the output file
        write_srt(translated_subtitles, args.output_file)
        print(f"Translation complete! Output saved to: {args.output_file}")
    
    except Exception as e:
        print(f"Error during translation: {e}")
        sys.exit(1)

def main():
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()