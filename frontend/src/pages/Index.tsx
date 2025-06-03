import React, { useState, useEffect } from 'react';
import { AgentSidebar } from '@/components/AgentSidebar';
import { ChatInterface } from '@/components/ChatInterface';
import { Agent, Conversation, AgentDefinition } from '@/types/Agent';

const Index = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  useEffect(() => {
    fetchAgents();
    fetchConversations();
  }, []);

  useEffect(() => {
    if (selectedAgent) {
      fetchConversations();
    }
  }, [selectedAgent]);

  const fetchAgents = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/agent-definitions`);
      const data = await response.json();
      setAgents(data);
      if (data.length > 0 && !selectedAgent) {
        setSelectedAgent(data[0]);
      }
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const fetchConversations = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/conversations`);
      const data = await response.json();
      setConversations(data);
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  const handleAgentCreate = async (agentData: Omit<AgentDefinition, 'id'>) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/agent-definitions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(agentData),
      });
      if (response.ok) {
        fetchAgents();
      }
    } catch (error) {
      console.error('Error creating agent:', error);
    }
  };

  const handleAgentUpdate = async (agentId: string, agentData: Partial<AgentDefinition>) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/agent-definitions/${agentId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(agentData),
      });
      if (response.ok) {
        fetchAgents();
      }
    } catch (error) {
      console.error('Error updating agent:', error);
    }
  };

  const handleAgentDelete = async (agentId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/agent-definitions/${agentId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        if (selectedAgent?.id === agentId) {
          setSelectedAgent(agents.find(a => a.id !== agentId) || null);
        }
        fetchAgents();
      }
    } catch (error) {
      console.error('Error deleting agent:', error);
    }
  };

  const handleNewConversation = async () => {
    if (!selectedAgent) return;
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/conversations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: selectedAgent.id }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setSelectedConversation(data.conversation_id);
        fetchConversations();
      }
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  const handleConversationDelete = async (conversationId: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/conversation/${conversationId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        if (selectedConversation === conversationId) {
          setSelectedConversation(null);
        }
        fetchConversations();
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };

  const handleAgentSelect = (agent: Agent) => {
    setSelectedAgent(agent);
    setSelectedConversation(null);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      <AgentSidebar
        agents={agents}
        selectedAgent={selectedAgent}
        onAgentSelect={handleAgentSelect}
        onAgentCreate={handleAgentCreate}
        onAgentUpdate={handleAgentUpdate}
        onAgentDelete={handleAgentDelete}
        conversations={conversations}
        selectedConversation={selectedConversation}
        onConversationSelect={setSelectedConversation}
        onConversationDelete={handleConversationDelete}
        onNewConversation={handleNewConversation}
        uploadedFile={uploadedFile}
        onFileUpload={setUploadedFile}
      />
      <div className="flex-1">
        <ChatInterface
          selectedAgent={selectedAgent}
          conversationId={selectedConversation}
          uploadedFile={uploadedFile}
          onFileUpload={setUploadedFile}
        />
      </div>
    </div>
  );
};

export default Index;
