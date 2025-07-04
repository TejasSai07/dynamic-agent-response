export interface AgentDefinition {
  id: string;
  name: string;
  description: string;
  system_prompt: string;
  model_type: string;
  memory_enabled: boolean;
  is_built_in: boolean;
  csv_upload_enabled?: boolean;
}

export interface Agent extends AgentDefinition {
  // Additional runtime properties if needed
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  reasoning_steps?: ReasoningStep[];
  plot_paths?: string[];
  code?: string;
  finalAnswer?: string;
  final_output?: {
    plot_paths?: string[];
    [key: string]: any;
  };
  isComplete?: boolean;
  output?: string;
  error?: string;
}


export interface ReasoningStep {
  step_number: number;
  reasoning: string;
  next_step: string;
  code: string;
  output: string;
  error: string;
  plot_path: string | null;
}

export interface Conversation {
  id: string;
  agent_type: string;
  agent_id?: string;
  label: string;
  latest_timestamp: string;
  created_at: string;
}
