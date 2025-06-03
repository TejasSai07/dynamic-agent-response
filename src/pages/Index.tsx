
import React, { useState, useEffect } from 'react';
import { AgentSidebar } from '@/components/AgentSidebar';
import { ChatInterface } from '@/components/ChatInterface';
import { Agent } from '@/types/Agent';

const Index = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [csvEnabled, setCsvEnabled] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await fetch('/get_agents');
      const data = await response.json();
      setAgents(data);
      if (data.length > 0 && !selectedAgent) {
        setSelectedAgent(data[0]);
      }
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const handleAgentCreate = async (agentData: Omit<Agent, 'id'>) => {
    try {
      const response = await fetch('/add_agent', {
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

  const handleAgentUpdate = async (agentId: string, agentData: Partial<Agent>) => {
    try {
      const response = await fetch('/update_agent', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: agentId, ...agentData }),
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
      const response = await fetch('/delete_agent', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: agentId }),
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

  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      <AgentSidebar
        agents={agents}
        selectedAgent={selectedAgent}
        onAgentSelect={setSelectedAgent}
        onAgentCreate={handleAgentCreate}
        onAgentUpdate={handleAgentUpdate}
        onAgentDelete={handleAgentDelete}
        csvEnabled={csvEnabled}
        onCsvToggle={setCsvEnabled}
        uploadedFiles={uploadedFiles}
        onFilesUpload={setUploadedFiles}
      />
      <div className="flex-1">
        <ChatInterface
          selectedAgent={selectedAgent}
          csvEnabled={csvEnabled}
          uploadedFiles={uploadedFiles}
        />
      </div>
    </div>
  );
};

export default Index;
