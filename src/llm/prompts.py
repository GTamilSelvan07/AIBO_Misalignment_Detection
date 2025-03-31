"""
Prompts for LLM interactions in the misalignment detection system.
"""
from typing import Dict, List, Optional


def get_misalignment_prompt(utterance: str, 
                          person_name: str, 
                          context: Optional[List[Dict]] = None) -> str:
    """
    Generate a prompt for detecting misalignment in speech.
    
    Args:
        utterance: The utterance to analyze
        person_name: Name of the person speaking
        context: Optional list of previous utterances
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Prepare context if provided
    context_text = ""
    if context and len(context) > 0:
        context_text = "Previous utterances:\n"
        for i, ctx in enumerate(context[-3:]):  # Use up to last 3 utterances
            speaker = "Other" if i % 2 == 0 else person_name
            context_text += f"{speaker}: {ctx['text']}\n"
    
    prompt = f"""You are analyzing a conversation to detect misalignment, confusion, or misunderstanding.

{context_text}

Current utterance from {person_name}: "{utterance}"

Task: Analyze this utterance for signs of confusion, misalignment, or misunderstanding.
1. Look for explicit statements of confusion ("I don't understand", "That's unclear", etc.)
2. Look for questioning phrases that suggest misalignment ("What do you mean?", "Could you explain?")
3. Look for speech patterns that indicate uncertainty (hesitations, filler words, etc.)
4. Consider the context of the conversation

Respond with a JSON object that includes:
1. A misalignment score from 0-100 (0=completely aligned, 100=completely confused)
2. A brief explanation of your reasoning
3. Specific phrases or words that indicate misalignment
4. The type of misalignment (conceptual confusion, request for clarity, etc.)

JSON format:
```json
{{
  "misalignment_score": <0-100>,
  "explanation": "<brief explanation>",
  "indicators": ["<phrase1>", "<phrase2>", ...],
  "misalignment_type": "<type of misalignment>",
  "confidence": <0-1>
}}
```
"""
    
    return prompt


def get_combined_analysis_prompt(visual_score: int, 
                               visual_details: Dict,
                               speech_score: int,
                               speech_details: Dict,
                               transcript: str) -> str:
    """
    Generate a prompt for analyzing combined misalignment evidence.
    
    Args:
        visual_score: Visual misalignment score
        visual_details: Visual misalignment details
        speech_score: Speech misalignment score
        speech_details: Speech misalignment details
        transcript: Transcript of the speech
        
    Returns:
        str: Formatted prompt for the LLM
    """
    prompt = f"""You are analyzing evidence of misalignment and confusion in a conversation.

VISUAL ANALYSIS:
- Misalignment Score: {visual_score}/100
- Active facial expressions: {', '.join([f"AU{au}" for au in visual_details.get('active_aus', {})])}

SPEECH ANALYSIS:
- Misalignment Score: {speech_score}/100
- Explanation: {speech_details.get('explanation', 'No explanation available')}
- Detected indicators: {', '.join(speech_details.get('llm_analysis', {}).get('indicators', []) if speech_details.get('llm_analysis') else ['None'])}

TRANSCRIPT:
"{transcript}"

Task: Provide a holistic analysis of the person's misalignment or confusion based on both visual and verbal cues.

Respond with a JSON object that includes:
1. A combined misalignment score from 0-100
2. An explanation that considers both visual and verbal indicators
3. The likely source or cause of any misalignment
4. Suggestions for how the other person could address this misalignment

JSON format:
```json
{{
  "combined_score": <0-100>,
  "explanation": "<comprehensive explanation>",
  "likely_cause": "<source of misalignment>",
  "suggestions": ["<suggestion1>", "<suggestion2>", ...],
  "confidence": <0-1>
}}
```
"""
    
    return prompt


def get_conversation_summary_prompt(conversation_history: List[Dict]) -> str:
    """
    Generate a prompt for summarizing a conversation's misalignment patterns.
    
    Args:
        conversation_history: List of utterances with misalignment scores
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Extract utterances and scores
    utterances_text = "\n".join([
        f"[Score: {u.get('score', 'N/A')}/100] {u.get('text', '')}"
        for u in conversation_history if u.get('text')
    ])
    
    prompt = f"""You are analyzing a conversation to identify patterns of misalignment and confusion.

CONVERSATION HISTORY WITH MISALIGNMENT SCORES:
{utterances_text}

Task: Analyze this conversation to identify patterns of misalignment and confusion.

Respond with a JSON object that includes:
1. A summary of misalignment patterns observed
2. Topics or concepts that appear to cause the most confusion
3. Suggestions for improving alignment in future conversations
4. An overall assessment of communication effectiveness

JSON format:
```json
{{
  "misalignment_patterns": ["<pattern1>", "<pattern2>", ...],
  "problematic_topics": ["<topic1>", "<topic2>", ...],
  "improvement_suggestions": ["<suggestion1>", "<suggestion2>", ...],
  "overall_assessment": "<brief assessment>",
  "confidence": <0-1>
}}
```
"""
    
    return prompt