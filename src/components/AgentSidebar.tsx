
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Plus, Settings, Trash2, MessageSquare, X } from 'lucide-react';
import { Agent, Conversation, AgentDefinition } from '@/types/Agent';
import { AgentForm } from './AgentForm';
import { FileUpload } from './FileUpload';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

interface AgentSidebarProps {
  agents: Agent[];
  selectedAgent: Agent | null;
  onAgentSelect: (agent: Agent) => void;
  onAgentCreate: (agent: Omit<AgentDefinition, 'id'>) => void;
  onAgentUpdate: (agentId: string, agent: Partial<AgentDefinition>) => void;
  onAgentDelete: (agentId: string) => void;
  conversations: Conversation[];
  selectedConversation: string | null;
  onConversationSelect: (conversationId: string) => void;
  onConversationDelete: (conversationId: string) => void;
  onNewConversation: () => void;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
}

export const AgentSidebar: React.FC<AgentSidebarProps> = ({
  agents,
  selectedAgent,
  onAgentSelect,
  onAgentCreate,
  onAgentUpdate,
  onAgentDelete,
  conversations,
  selectedConversation,
  onConversationSelect,
  onConversationDelete,
  onNewConversation,
  uploadedFile,
  onFileUpload,
}) => {
  const [showForm, setShowForm] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);

  const handleCreateAgent = () => {
    setEditingAgent(null);
    setShowForm(true);
  };

  const handleEditAgent = (agent: Agent) => {
    if (agent.is_built_in) return; // Can't edit built-in agents
    setEditingAgent(agent);
    setShowForm(true);
  };

  const handleFormSubmit = (agentData: Omit<AgentDefinition, 'id'>) => {
    if (editingAgent) {
      onAgentUpdate(editingAgent.id, agentData);
    } else {
      onAgentCreate(agentData);
    }
    setShowForm(false);
    setEditingAgent(null);
  };

  const handleDeleteConversation = async (conversationId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      try {
        const response = await fetch(`${API_URL}/conversation/${conversationId}`, {
          method: 'DELETE',
        });
        if (response.ok) {
          onConversationDelete(conversationId);
        }
      } catch (error) {
        console.error('Error deleting conversation:', error);
      }
    }
  };

  // Filter conversations for the selected agent
  const agentConversations = selectedAgent 
    ? conversations.filter(conv => {
        // For built-in agents, match by agent_type
        if (selectedAgent.is_built_in) {
          return conv.agent_type === selectedAgent.id;
        }
        // For custom agents, check if conversation belongs to this agent
        return conv.agent_type === 'custom';
      })
    : [];

  return (
    <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
      <div className="p-4">
        <h1 className="text-xl font-bold mb-4">AI Agent Manager</h1>
        <Button
          onClick={handleCreateAgent}
          className="w-full mb-4 bg-blue-600 hover:bg-blue-700"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create Agent
        </Button>
        
        {selectedAgent && (
          <Button
            onClick={onNewConversation}
            className="w-full mb-4 bg-green-600 hover:bg-green-700"
          >
            <MessageSquare className="w-4 h-4 mr-2" />
            New Conversation
          </Button>
        )}
      </div>

      <Separator className="bg-gray-700" />

      <div className="p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">AGENTS</h3>
        <ScrollArea className="h-48">
          <div className="space-y-2">
            {agents.map((agent) => (
              <div
                key={agent.id}
                className={`p-3 rounded-lg cursor-pointer transition-all duration-200 group ${
                  selectedAgent?.id === agent.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
                onClick={() => onAgentSelect(agent)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="font-medium">{agent.name}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {agent.model_type} {agent.is_built_in ? '• Built-in' : '• Custom'}
                    </div>
                  </div>
                  {!agent.is_built_in && (
                    <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditAgent(agent);
                        }}
                        className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                      >
                        <Settings className="w-3 h-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          onAgentDelete(agent.id);
                        }}
                        className="h-6 w-6 p-0 text-gray-400 hover:text-red-400"
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      <Separator className="bg-gray-700" />

      <div className="p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          {selectedAgent ? `${selectedAgent.name.toUpperCase()} CONVERSATIONS` : 'SELECT AGENT'}
        </h3>
        {selectedAgent ? (
          <ScrollArea className="h-32">
            <div className="space-y-2">
              {agentConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`p-2 rounded-lg cursor-pointer transition-all duration-200 group flex items-center justify-between ${
                    selectedConversation === conversation.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                  onClick={() => onConversationSelect(conversation.id)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{conversation.label}</div>
                    <div className="text-xs text-gray-400">
                      {new Date(conversation.latest_timestamp).toLocaleDateString()}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                    className="h-6 w-6 p-0 text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              ))}
              {agentConversations.length === 0 && (
                <div className="text-sm text-gray-400 text-center py-4">
                  No conversations yet. Start a new one!
                </div>
              )}
            </div>
          </ScrollArea>
        ) : (
          <div className="text-sm text-gray-400 text-center py-4">
            Select an agent to view its conversations
          </div>
        )}
      </div>

      <Separator className="bg-gray-700" />

      <div className="p-4 flex-1">
        <FileUpload
          uploadedFile={uploadedFile}
          onFileUpload={onFileUpload}
        />
      </div>

      {showForm && (
        <AgentForm
          agent={editingAgent}
          onSubmit={handleFormSubmit}
          onCancel={() => {
            setShowForm(false);
            setEditingAgent(null);
          }}
        />
      )}
    </div>
  );
};
