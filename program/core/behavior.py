"""
Behavior management module
Manages AI desktop pet states and behavior transitions
"""
import random
from enum import Enum
from collections import deque


class State(Enum):
    """AI desktop pet state enumeration"""
    IDLE = "idle"
    THINKING = "thinking"
    SLEEP = "sleep"
    TALK = "talk"
    WAVE = "wave"


class BehaviorManager:
    """Behavior manager"""
    
    def __init__(self):
        """Initialize behavior manager"""
        self.current_state = State.IDLE
        self.target_state = None
        self.state_duration = 0.0
        self.total_time = 0.0
        self.idle_time = 0.0
        self.is_animation_playing = False
        self._need_restart_animation = False
        
        self.manual_control_enabled = False
        self.next_state_input = None
        self.waiting_for_input = False
        self.state_queue = deque()
        
        self.state_durations = {
            State.IDLE: (2.0, 5.0),
            State.THINKING: (1.5, 2.5),
            State.TALK: (2.0, 3.0),
        }
        
        self.transition_rules = {
            State.IDLE: {
                "mouse_move": State.THINKING,
                "click": State.TALK,
                "auto_move": State.THINKING,
            },
            State.THINKING: {
                "mouse_move": State.IDLE,
                "done": State.IDLE,
            },
            State.TALK: {
                "talk_command": State.IDLE,
                "done": State.IDLE,
            },
        }
        
        self.pending_transition = None
        self._reset_state_duration()
        self.long_idle_threshold = 10.0
    
    def _get_random_duration(self, state):
        """Get random duration for state"""
        min_dur, max_dur = self.state_durations[state]
        return random.uniform(min_dur, max_dur)
    
    def _reset_state_duration(self):
        """Reset state duration"""
        min_dur, max_dur = self.state_durations[self.current_state]
        self.target_duration = random.uniform(min_dur, max_dur)
        self.is_animation_playing = True
        self.state_duration = 0.0
    
    def update(self, delta_time):
        """
        Update behavior state
        
        Args:
            delta_time: Time delta (seconds)
        """
        self.total_time += delta_time
        self.state_duration += delta_time
        
        if self.current_state == State.IDLE:
            self.idle_time += delta_time
        else:
            self.idle_time = 0.0
        
        if self.state_duration >= self.target_duration:
            self.is_animation_playing = False
            
            if self.target_state is not None:
                next_state = self.target_state
                self.target_state = None
                self._transition_to_state(next_state)
                return
            
            if len(self.state_queue) > 0:
                next_state = self.state_queue.popleft()
                self._transition_to_state(next_state)
                return
            
            self._restart_current_animation()
    
    def request_transition(self, trigger_event):
        """
        Request state transition (deprecated, use set_target_state)
        
        Args:
            trigger_event: Trigger event name
        
        Returns:
            bool: Whether transition was successfully requested
        """
        rules = self.transition_rules.get(self.current_state, {})
        if trigger_event in rules:
            next_state = rules[trigger_event]
            self.set_target_state(next_state)
            return True
        return False
    
    def set_target_state(self, target_state):
        """
        Set target state
        
        Args:
            target_state: Target state (State enum or string)
        """
        if isinstance(target_state, str):
            state_map = {
                "idle": State.IDLE,
                "talk": State.TALK,
                "thinking": State.THINKING,
            }
            target_state = state_map.get(target_state.lower().strip())
            if target_state is None:
                return False
        
        if target_state == self.current_state:
            self.target_state = None
            return True
        
        self.target_state = target_state
        
        if not self.is_animation_playing:
            self._transition_to_state(target_state)
            self.target_state = None
        return True
    
    def _restart_current_animation(self):
        """Restart current state animation"""
        self._reset_state_duration()
        self._need_restart_animation = True
    
    def _transition_to_state(self, next_state):
        """
        Transition to specified state
        
        Args:
            next_state: Target state (State enum)
        """
        if next_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = next_state
        self.target_state = None
        self._reset_state_duration()
        
        if self.manual_control_enabled:
            self.waiting_for_input = False
    
    def _auto_transition(self):
        """Auto state transition"""
        if self.pending_transition:
            trigger = self.pending_transition
            self.pending_transition = None
            if self.request_transition(trigger):
                return
        
        if self.current_state == State.THINKING:
            self.set_target_state(State.IDLE)
        elif self.current_state == State.TALK:
            self.set_target_state(State.IDLE)
        elif self.current_state == State.IDLE:
            rand = random.random()
            if rand < 0.3:
                self.set_target_state(State.THINKING)
    
    def get_current_state(self):
        """
        Get current state
        
        Returns:
            State: Current state enum
        """
        return self.current_state
    
    def get_state_info(self):
        """
        Get state information
        
        Returns:
            dict: Dictionary containing state information
        """
        return {
            "state": self.current_state.value,
            "target_state": self.target_state.value if self.target_state else None,
            "duration": self.state_duration,
            "target_duration": self.target_duration,
            "total_time": self.total_time,
            "idle_time": self.idle_time,
            "is_animation_playing": self.is_animation_playing,
            "pending_transition": self.pending_transition,
            "queue_length": len(self.state_queue),
            "queue": self.get_queue_info()
        }
    
    def can_transition(self):
        """Check if state transition is possible"""
        return True
    
    def is_animation_running(self):
        """Check if animation is playing"""
        return self.is_animation_playing
    
    def enable_manual_control(self):
        """Enable manual control mode"""
        self.manual_control_enabled = True
        self.waiting_for_input = False
    
    def disable_manual_control(self):
        """Disable manual control mode"""
        self.manual_control_enabled = False
        self.waiting_for_input = False
        self.next_state_input = None
    
    def set_next_state(self, state_str):
        """
        Set next state
        
        Args:
            state_str: State string ("idle", "talk", "thinking")
        
        Returns:
            bool: Whether state was successfully set
        """
        state_map = {
            "idle": State.IDLE,
            "talk": State.TALK,
            "thinking": State.THINKING,
        }
        
        state_str_lower = state_str.lower().strip()
        if state_str_lower not in state_map:
            return False
        
        next_state = state_map[state_str_lower]
        return self.set_target_state(next_state)
    
    def get_queue_length(self):
        """Get state queue length"""
        return len(self.state_queue)
    
    def clear_queue(self):
        """Clear state queue"""
        self.state_queue.clear()
    
    def get_queue_info(self):
        """Get queue information"""
        return [state.value for state in self.state_queue]
