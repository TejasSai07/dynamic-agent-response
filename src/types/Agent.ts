
export interface Agent {
  id: string;
  name: string;
  description: string;
  model_type: string;
  memory_enabled?: boolean;
  is_built_in: boolean;
}

export interface Conversation {
  id: string;
  agent_type: string;
  label: string;
  latest_timestamp: string;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  code?: string;
  output?: string;
  error?: string;
  plot_path?: string;
  plot_paths?: string[];
  tool_used?: string;
  tool_payload?: any;
  tool_output?: any;
  step_number?: number;
  reasoning_steps?: ReasoningStep[];
  has_pickled_objects?: boolean;
}

export interface ReasoningStep {
  step_number: number;
  reasoning: string;
  next_step: string;
  code: string;
  output?: string;
  error?: string;
  plot_path?: string;
}

export interface AgentDefinition {
  id: string;
  name: string;
  model_type: string;
  system_prompt: string;
  tools: string[];
  memory_enabled: boolean;
  tasks: string[];
}
