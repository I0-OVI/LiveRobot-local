"""
Importance calculator module
Uses LLM to evaluate memory importance across multiple dimensions
"""
import json
import re
from typing import Dict, Optional, Tuple


class ImportanceCalculator:
    """Calculates memory importance using LLM evaluation"""
    
    PROMPT_TEMPLATE = """You are a memory importance evaluator.
Given a candidate memory, score the following from 0 to 1:

1. Semantic reusability
   Is this information reusable across future conversations?

2. Temporal stability
   How likely is this information to remain valid after 1 week or longer period of time?

3. User binding
   Is this information very personal and unique for the user?

4. Decision impact
   Would this information change future agent decisions?

5. Noise level
   How much noise or irrelevant information is in this memory? (Lower is better, so 0 = no noise, 1 = high noise)

Return JSON only with the following format:
{{
    "semantic_reusability": 0.0-1.0,
    "temporal_stability": 0.0-1.0,
    "user_binding": 0.0-1.0,
    "decision_impact": 0.0-1.0,
    "noise_level": 0.0-1.0
}}

Memory to evaluate:
User: {user_input}
Assistant: {assistant_response}

JSON:"""
    
    def __init__(self, text_generator=None):
        """
        Initialize importance calculator
        
        Args:
            text_generator: QwenTextGenerator instance for LLM evaluation
        """
        self.text_generator = text_generator
    
    def calculate_importance(
        self,
        user_input: str,
        assistant_response: str,
        fallback_importance: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate importance score using LLM
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
            fallback_importance: Fallback importance if calculation fails
        
        Returns:
            Tuple of (overall_importance_score, detailed_scores_dict)
        """
        if not self.text_generator:
            print("[ImportanceCalculator] Warning: No text generator available, using fallback")
            return fallback_importance, {}
        
        if not user_input and not assistant_response:
            return 0.0, {}
        
        try:
            # Build prompt
            prompt = self.PROMPT_TEMPLATE.format(
                user_input=user_input,
                assistant_response=assistant_response
            )
            
            # Generate evaluation using LLM
            # Use chat method with empty history for importance evaluation
            if hasattr(self.text_generator, 'chat'):
                # Use chat method with empty history
                response, _ = self.text_generator.chat(prompt, history=[])
            elif hasattr(self.text_generator, 'generate_simple'):
                response = self.text_generator.generate_simple(prompt)
            elif hasattr(self.text_generator, 'generate_text'):
                response = self.text_generator.generate_text(prompt, max_new_tokens=200)
            else:
                print("[ImportanceCalculator] Warning: Text generator doesn't support generation")
                return fallback_importance, {}
            
            # Parse JSON from response
            scores = self._parse_json_response(response)
            
            if scores:
                # Calculate overall importance
                # Formula: weighted average, with noise_level inverted (lower noise = higher importance)
                overall = self._calculate_overall_importance(scores)
                return overall, scores
            else:
                print(f"[ImportanceCalculator] Warning: Failed to parse scores, using fallback")
                return fallback_importance, {}
                
        except Exception as e:
            print(f"[ImportanceCalculator] Error calculating importance: {e}")
            import traceback
            traceback.print_exc()
            return fallback_importance, {}
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, float]]:
        """
        Parse JSON from LLM response
        
        Args:
            response: LLM response text
        
        Returns:
            Dictionary with scores, or None if parsing fails
        """
        if not response:
            return None
        
        # Try to extract JSON from response
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove code block markers
            response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.MULTILINE)
            response = re.sub(r'\s*```\s*$', '', response, flags=re.MULTILINE)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        
        try:
            scores = json.loads(json_str)
            
            # Validate and normalize scores
            required_keys = [
                'semantic_reusability',
                'temporal_stability',
                'user_binding',
                'decision_impact',
                'noise_level'
            ]
            
            result = {}
            for key in required_keys:
                value = scores.get(key, 0.5)
                # Ensure value is between 0 and 1
                value = max(0.0, min(1.0, float(value)))
                result[key] = value
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ImportanceCalculator] JSON parse error: {e}")
            print(f"[ImportanceCalculator] Response was: {response[:200]}")
            return None
        except Exception as e:
            print(f"[ImportanceCalculator] Error parsing scores: {e}")
            return None
    
    def _calculate_overall_importance(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall importance from detailed scores
        
        Args:
            scores: Dictionary with detailed scores
        
        Returns:
            Overall importance score (0.0-1.0)
        """
        # Weighted average formula with specified weights
        weights = {
            'semantic_reusability': 0.30,
            'temporal_stability': 0.25,
            'user_binding': 0.20,
            'decision_impact': 0.15,
            'noise_level': 0.10  # Inverted in calculation (noise_penalty)
        }
        
        # Calculate weighted average
        # Note: noise_level is inverted (1 - noise_level) because lower noise = better
        overall = (
            scores.get('semantic_reusability', 0.5) * weights['semantic_reusability'] +
            scores.get('temporal_stability', 0.5) * weights['temporal_stability'] +
            scores.get('user_binding', 0.5) * weights['user_binding'] +
            scores.get('decision_impact', 0.5) * weights['decision_impact'] +
            (1.0 - scores.get('noise_level', 0.5)) * weights['noise_level']  # Invert noise (noise_penalty)
        )
        
        # Ensure between 0 and 1
        return max(0.0, min(1.0, overall))
    
    def set_text_generator(self, text_generator):
        """
        Set text generator for importance calculation
        
        Args:
            text_generator: QwenTextGenerator instance
        """
        self.text_generator = text_generator
