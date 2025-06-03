
export interface Agent {
  id: string;
  name: string;
  model: string;
  prompt: string;
  tools: string[];
  enable_memory: boolean;
  memory_data?: any;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  code?: string;
  graph?: string;
  finalAnswer?: string;
}

export interface AgentResponse {
  code: string;
  graph: string;
  final_answer: string;
}
