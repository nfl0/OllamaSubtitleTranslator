#!/usr/bin/env python3
"""
Subtitle Translator CLI

A command-line tool for translating subtitle files using LLMs through Ollama.
This application provides context-aware translation to maintain consistency
across subtitle sequences.
"""

import sys
import re
import os
import subprocess
import time
import argparse
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio

# LangChain imports
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from pydantic import BaseModel, Field

__version__ = "1.0.0"

class SubtitleEntry(BaseModel):
    """Model for a subtitle entry."""
    index: int = Field(description="The subtitle index number")
    timing: str = Field(description="The timing information")
    text: str = Field(description="The subtitle text")

class SubtitleBatch(BaseModel):
    """Model for a batch of translated subtitles."""
    entries: List[SubtitleEntry] = Field(description="List of subtitle entries")

class SubtitleParser:
    """Parser for subtitle files."""
    
    @staticmethod
    def parse_srt(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse an SRT subtitle file into a list of subtitle entries.
        
        Args:
            file_path: Path to the SRT file
            
        Returns:
            List of subtitle dictionaries with index, timing, and text
        """
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
    
    @staticmethod
    def write_srt(subtitles: List[Dict[str, Any]], output_path: str) -> None:
        """
        Write subtitles back to SRT format.
        
        Args:
            subtitles: List of subtitle dictionaries
            output_path: Path to write the output file
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            for i, sub in enumerate(subtitles):
                if i > 0:
                    file.write('\n\n')
                file.write(f"{sub['index']}\n")
                file.write(f"{sub['timing']}\n")
                file.write(sub['text'])

class SubtitleProcessor:
    """Processes subtitles for translation."""
    
    @staticmethod
    def clean_subtitle_text(text: str) -> str:
        """
        Remove any commentary or notes from the subtitle text.
        
        Args:
            text: Original subtitle text
            
        Returns:
            Cleaned subtitle text
        """
        # Remove lines starting with * or **
        lines = text.split('\n')
        clean_lines = [line for line in lines if not line.strip().startswith('*')]
        
        # Remove content after "**Key Changes:" or similar markers
        clean_text = '\n'.join(clean_lines)
        commentary_markers = ["**Key Changes:", "**Note:", "**Translation note:", "**Character note:"]
        for marker in commentary_markers:
            if marker in clean_text:
                clean_text = clean_text.split(marker)[0].strip()
        
        return clean_text
    
    @staticmethod
    def batch_subtitles(subtitles: List[Dict[str, Any]], batch_size: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Group subtitles into batches for more efficient translation.
        
        Args:
            subtitles: List of subtitle dictionaries
            batch_size: Maximum number of subtitles per batch
            
        Returns:
            List of batches, where each batch is a list of subtitle dictionaries
        """
        batches = []
        current_batch = []
        current_batch_text_length = 0
        max_chars_per_batch = 1000  # Limit characters per batch
        
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

class TranslationEngine:
    """Engine for translating subtitles using LLMs."""
    
    def __init__(self, model_name: str, temperature: float):
        """
        Initialize the translation engine.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for translation (higher = more creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
        )
    
    async def translate_batch_with_context(
        self, 
        batch: List[Dict[str, Any]], 
        source_lang: str, 
        target_lang: str, 
        all_batches: Optional[List[List[Dict[str, Any]]]] = None, 
        current_batch_index: Optional[int] = None, 
        metadata: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of subtitles using LangChain with context from surrounding batches.
        
        Args:
            batch: List of subtitle dictionaries to translate
            source_lang: Source language
            target_lang: Target language
            all_batches: All batches for context
            current_batch_index: Index of current batch in all_batches
            metadata: Optional metadata about the content
            
        Returns:
            Translated batch of subtitles
        """
        # Create input for the model
        subtitle_texts = []
        for sub in batch:
            subtitle_texts.append(f"[{sub['index']}] {sub['text']}")
        
        combined_text = "\n\n".join(subtitle_texts)
        
        # Add context if available
        context = ""
        if all_batches and current_batch_index is not None:
            # Add previous batch for context if available
            if current_batch_index > 0:
                prev_batch = all_batches[current_batch_index - 1]
                context += "PREVIOUS CONTEXT (not to be translated, only for reference):\n"
                for sub in prev_batch:
                    context += f"[{sub['index']}] {sub['text']}\n"
                context += "\n"
            
            # Add next batch for context if available
            if current_batch_index < len(all_batches) - 1:
                next_batch = all_batches[current_batch_index + 1]
                context += "FOLLOWING CONTEXT (not to be translated, only for reference):\n"
                for sub in next_batch:
                    context += f"[{sub['index']}] {sub['text']}\n"
                context += "\n"
        
        # Create metadata section if available
        metadata_text = ""
        if metadata:
            metadata_text = f"""CONTENT METADATA:
This content is: {metadata}
Adapt your translation to maintain the appropriate tone, terminology, and style for this specific content.

"""
        
        # Updated translation prompt with more explicit instructions
        source_lang_prompt = f"from {source_lang}" if source_lang.lower() != "auto" else ""
        template = f"""{metadata_text}Translate the following subtitles {source_lang_prompt} to {target_lang}.
IMPORTANT: Your response must ONLY contain translated subtitles in the exact format requested.
DO NOT include any notes, commentary, or thinking process in your response.
DO NOT use any prefixes like "Translation:" or "**" before the subtitles.
Each subtitle should only be preceded by its index number in square brackets.

Preserve the original meaning, formatting, and line breaks as accurately as possible.
These subtitles are sequential dialogue from the same content, so maintain consistency in:
- Character names and how they refer to each other
- Terminology specific to the content
- Tone and style of speech for each character
- References to events or objects mentioned earlier

{context}

SUBTITLES TO TRANSLATE:
{{text}}

TRANSLATION FORMAT EXAMPLE:
[1] This is the first translated subtitle.
[2] This is the second translated subtitle.

YOUR TRANSLATION:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", template)
        ])
        
        # Create the chain
        chain = prompt | self.llm
        
        # Execute the chain
        result = await chain.ainvoke({"text": combined_text})
        
        # Parse the results
        translated_text = result.content if hasattr(result, 'content') else str(result)
        
        # Improved pattern to extract only the subtitle translations
        # This pattern focuses on matching the exact [index] format followed by text
        pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)'
        matches = re.finditer(pattern, translated_text, re.DOTALL)
        
        # Create a dict to map indexes to translations
        translations = {}
        for match in matches:
            sub_index = int(match.group(1))
            sub_text = match.group(2).strip()
            translations[sub_index] = sub_text
        
        # Update the batch with translations
        for sub in batch:
            if sub['index'] in translations:
                sub['text'] = translations[sub['index']]
        
        return batch
    
    async def translate_individual_with_context(
        self, 
        subtitle: Dict[str, Any], 
        source_lang: str, 
        target_lang: str, 
        context: Optional[Dict[str, str]] = None, 
        metadata: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate a single subtitle using LangChain with context.
        
        Args:
            subtitle: Subtitle dictionary to translate
            source_lang: Source language
            target_lang: Target language
            context: Context from previous and next subtitles
            metadata: Optional metadata about the content
            
        Returns:
            Translated subtitle
        """
        # Create context section for the prompt
        context_text = ""
        if context:
            if "previous" in context:
                context_text += f"PREVIOUS CONTEXT (not to be translated, only for reference):\n{context['previous']}\n\n"
            if "next" in context:
                context_text += f"FOLLOWING CONTEXT (not to be translated, only for reference):\n{context['next']}\n\n"
        
        # Create metadata section if available
        metadata_text = ""
        if metadata:
            metadata_text = f"""CONTENT METADATA:
This content is: {metadata}
Adapt your translation to maintain the appropriate tone, terminology, and style for this specific content.

"""
        
        # Create translation prompt with clearer instructions
        source_lang_prompt = f"from {source_lang}" if source_lang.lower() != "auto" else ""
        template = f"""{metadata_text}Translate the following subtitle {source_lang_prompt} to {target_lang}.
IMPORTANT: Your response must ONLY contain the translated subtitle text.
DO NOT include any notes, commentary, thinking process, or formatting.
DO NOT include the index number in your response.
DO NOT use any prefixes like "Translation:" or "**" before the subtitle.

Preserve the original meaning, formatting, and line breaks as accurately as possible.
Consider the context provided to ensure natural dialogue flow.

{context_text}

SUBTITLE TO TRANSLATE:
{{text}}

TRANSLATED TEXT ONLY:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", template)
        ])
        
        # Create the chain
        chain = prompt | self.llm
        
        # Execute the chain
        result = await chain.ainvoke({"text": subtitle['text']})
        
        # Get the translation and clean it up
        translated_text = result.content if hasattr(result, 'content') else str(result)
        
        # Remove any line number indicators or other prefixes that might remain
        cleaned_text = re.sub(r'^\s*\*+\s*', '', translated_text)  # Remove ** prefixes
        cleaned_text = re.sub(r'^\s*Translation:\s*', '', cleaned_text)  # Remove "Translation:" prefix
        cleaned_text = re.sub(r'^\s*\[\d+\]\s*', '', cleaned_text)  # Remove [index] if present
        
        # Update the subtitle
        subtitle['text'] = cleaned_text.strip()
        
        return subtitle
    
    async def improve_translation_consistency(
        self, 
        translated_subtitles: List[Dict[str, Any]], 
        target_lang: str, 
        metadata: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Second pass to improve consistency across all translations.
        
        Args:
            translated_subtitles: List of translated subtitle dictionaries
            target_lang: Target language
            metadata: Optional metadata about the content
            
        Returns:
            Improved subtitles
        """
        # Group subtitles into manageable chunks for review
        chunk_size = 20
        chunks = [translated_subtitles[i:i+chunk_size] for i in range(0, len(translated_subtitles), chunk_size)]
        
        improved_subtitles = []
        
        # Create metadata section if available
        metadata_text = ""
        if metadata:
            metadata_text = f"""CONTENT METADATA:
This content is: {metadata}
Ensure your improvements maintain the appropriate tone, terminology, and style for this specific content.

"""
        
        for chunk in tqdm_asyncio(chunks, desc="Improving consistency"):
            # Extract text with indices
            chunk_text = ""
            for sub in chunk:
                chunk_text += f"[{sub['index']}] {sub['text']}\n\n"
            
            template = f"""{metadata_text}Review and improve the following translated subtitles in {target_lang}.
IMPORTANT: Your response must ONLY contain the improved subtitle texts.
DO NOT include any notes, commentary, or thinking process in your response.
Each subtitle should only be preceded by its index number in square brackets.

Focus on maintaining consistency in terminology, character references, and narrative flow.
Preserve the original meaning and formatting, but make the dialogue flow more naturally.
Do not change the subtitle index numbers in square brackets.

TRANSLATED SUBTITLES:
{{text}}

RESPONSE FORMAT EXAMPLE:
[1] Improved subtitle text.
[2] Another improved subtitle text.

IMPROVED TRANSLATIONS:"""
            
            prompt = ChatPromptTemplate.from_messages([("human", template)])
            chain = prompt | self.llm
            
            # Execute the chain
            result = await chain.ainvoke({"text": chunk_text})
            
            # Process the improved translations
            improved_text = result.content if hasattr(result, 'content') else str(result)
            
            # Parse the improved text with more robust pattern matching
            pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)'
            matches = re.finditer(pattern, improved_text, re.DOTALL)
            
            # Create a map for the improved subtitles
            improved_map = {}
            for match in matches:
                sub_index = int(match.group(1))
                sub_text = match.group(2).strip()
                
                # Clean up the text to remove any potential commentary
                sub_text = re.sub(r'^\s*\*+\s*', '', sub_text)  # Remove ** prefixes
                sub_text = re.sub(r'^\s*-\s*', '', sub_text)    # Remove leading dashes
                
                improved_map[sub_index] = sub_text
            
            # Update the chunk with improvements
            processed_chunk = []
            for sub in chunk:
                sub_copy = sub.copy()
                if sub['index'] in improved_map:
                    sub_copy['text'] = improved_map[sub['index']]
                processed_chunk.append(sub_copy)
            
            improved_subtitles.extend(processed_chunk)
        
        return improved_subtitles
    
    def cleanup(self):
        """Clean up resources."""
        try:
            subprocess.run(["ollama", "stop", self.model_name], check=False)
        except Exception as e:
            print(f"Note: Could not stop model: {e}")

class SubtitleTranslator:
    """Main class for subtitle translation."""
    
    def __init__(self):
        """Initialize the subtitle translator."""
        self.parser = SubtitleParser()
        self.processor = SubtitleProcessor()
    
    async def process_subtitles(
        self, 
        subtitles: List[Dict[str, Any]], 
        batch_size: int, 
        individual: bool, 
        source_lang: str, 
        target_lang: str, 
        model_name: str, 
        temperature: float, 
        enable_consistency_pass: bool = True, 
        metadata: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all subtitles with appropriate method.
        
        Args:
            subtitles: List of subtitle dictionaries
            batch_size: Maximum number of subtitles per batch
            individual: Whether to process subtitles individually
            source_lang: Source language
            target_lang: Target language
            model_name: Ollama model name
            temperature: Temperature for translation
            enable_consistency_pass: Whether to perform a consistency pass
            metadata: Optional metadata about the content
            
        Returns:
            Translated subtitles
        """
        # Initialize the translation engine
        engine = TranslationEngine(model_name, temperature)
        
        try:
            translated_subtitles = []
            
            if individual:
                # Process subtitles individually with context
                for i, sub in enumerate(tqdm_asyncio(subtitles, desc="Translating subtitles")):
                    # Add context from previous and next subtitles
                    context = {}
                    if i > 0:
                        context["previous"] = subtitles[i-1]["text"]
                    if i < len(subtitles) - 1:
                        context["next"] = subtitles[i+1]["text"]
                    
                    translated_sub = await engine.translate_individual_with_context(
                        sub.copy(), source_lang, target_lang, context, metadata
                    )
                    translated_subtitles.append(translated_sub)
                    # Add a small delay to avoid overwhelming Ollama
                    await asyncio.sleep(0.05)
            else:
                # Process in batches with context
                batches = self.processor.batch_subtitles(subtitles, batch_size)
                print(f"Processing in {len(batches)} batches")
                
                for i, batch in enumerate(tqdm_asyncio(batches, desc="Translating batches")):
                    try:
                        # Create copies of the subtitles to avoid modifying the originals
                        batch_copy = [sub.copy() for sub in batch]
                        translated_batch = await engine.translate_batch_with_context(
                            batch_copy, source_lang, target_lang, 
                            all_batches=batches, current_batch_index=i, metadata=metadata
                        )
                        translated_subtitles.extend(translated_batch)
                        # Add a small delay between batches
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        print(f"\nError in batch translation: {e}")
                        print("Falling back to individual translation for this batch")
                        
                        # Fall back to individual translation
                        for sub in batch:
                            try:
                                sub_copy = sub.copy()
                                # Add context from the batch
                                context = {}
                                sub_idx = batch.index(sub)
                                if sub_idx > 0:
                                    context["previous"] = batch[sub_idx-1]["text"]
                                if sub_idx < len(batch) - 1:
                                    context["next"] = batch[sub_idx+1]["text"]
                                    
                                translated_sub = await engine.translate_individual_with_context(
                                    sub_copy, source_lang, target_lang, context, metadata
                                )
                                translated_subtitles.append(translated_sub)
                                await asyncio.sleep(0.05)
                            except Exception as sub_e:
                                print(f"Error translating subtitle {sub['index']}: {sub_e}")
                                # Keep original if translation fails
                                translated_subtitles.append(sub.copy())
            
            # Second pass: improve consistency if enabled
            if enable_consistency_pass:
                print("\nStarting consistency improvement pass...")
                improved_subtitles = await engine.improve_translation_consistency(
                    translated_subtitles, target_lang, metadata
                )
                return improved_subtitles
            else:
                return translated_subtitles
                
        finally:
            # Clean up resources
            engine.cleanup()
    
    @staticmethod
    def ensure_ollama_serve():
        """
        Check if ollama serve is running.
        Exits if not running.
        """
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

def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='SubtitleTranslator: Translate SRT subtitle files using LLMs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('file', help='SRT file to translate')
    
    parser.add_argument(
        '-i', '--input-language', 
        default='Auto', 
        help='Source language (Auto for auto-detect)'
    )
    
    parser.add_argument(
        '-o', '--output-language', 
        default='English', 
        help='Target language'
    )
    
    parser.add_argument(
        '-m', '--model', 
        default='vanilj/gemma-2-ataraxy-9b:IQ2_M', 
        help='Ollama model to use'
    )
    
    parser.add_argument(
        '-f', '--output-file', 
        help='Output file name (default: input_file_translated.srt)'
    )
    
    parser.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=5,
        help='Number of subtitles to translate at once'
    )
    
    parser.add_argument(
        '-t', '--temperature', 
        type=float, 
        default=0.3,
        help='Model temperature for translation'
    )
    
    parser.add_argument(
        '--individual', 
        action='store_true',
        help='Translate each subtitle individually (slower but potentially more accurate)'
    )
    
    parser.add_argument(
        '--skip-consistency', 
        action='store_true',
        help='Skip the consistency improvement pass'
    )
    
    parser.add_argument(
        '--meta', '--metadata', 
        help='Metadata about the content (e.g., "The Walking Dead, Season 1, Episode 3")'
    )
    
    parser.add_argument(
        '-v', '--version', 
        action='version', 
        version=f'SubtitleTranslator v{__version__}'
    )
    
    return parser

async def main_async():
    """Async entry point for the application."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Set default output filename if not specified
    if not args.output_file:
        base_name, ext = os.path.splitext(args.file)
        args.output_file = f"{base_name}_{args.output_language.lower()}{ext}"
    
    translator = SubtitleTranslator()
    translator.ensure_ollama_serve()

    print(f"Parsing subtitle file: {args.file}")
    subtitles = translator.parser.parse_srt(args.file)
    print(f"Found {len(subtitles)} subtitle entries")
    
    print(f"Translating from {args.input_language} to {args.output_language} using {args.model}...")
    print(f"Context-enhanced translation enabled with {'individual' if args.individual else 'batch'} processing")
    if args.meta:
        print(f"Content metadata: {args.meta}")
    if not args.skip_consistency:
        print("Consistency improvement pass will be performed after initial translation")
    
    # Process all subtitles
    try:
        translated_subtitles = await translator.process_subtitles(
            subtitles, 
            args.batch_size, 
            args.individual, 
            args.input_language, 
            args.output_language, 
            args.model,
            args.temperature,
            not args.skip_consistency,
            args.meta
        )
        
        # Write the translated subtitles to the output file
        translator.parser.write_srt(translated_subtitles, args.output_file)
        print(f"Translation complete! Output saved to: {args.output_file}")
    
    except Exception as e:
        print(f"Error during translation: {e}")
        sys.exit(1)

def main():
    """Main entry point for the application."""
    try:
        # Run the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)

if __name__ == "__main__":
    main()