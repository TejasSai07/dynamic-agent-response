
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Plus, Upload, Settings, Trash2 } from 'lucide-react';
import { Agent } from '@/types/Agent';
import { AgentForm } from './AgentForm';
import { CSVUpload } from './CSVUpload';

interface AgentSidebarProps {
  agents: Agent[];
  selectedAgent: Agent | null;
  onAgentSelect: (agent: Agent) => void;
  onAgentCreate: (agent: Omit<Agent, 'id'>) => void;
  onAgentUpdate: (agentId: string, agent: Partial<Agent>) => void;
  onAgentDelete: (agentId: string) => void;
  csvEnabled: boolean;
  onCsvToggle: (enabled: boolean) => void;
  uploadedFiles: File[];
  onFilesUpload: (files: File[]) => void;
}

export const AgentSidebar: React.FC<AgentSidebarProps> = ({
  agents,
  selectedAgent,
  onAgentSelect,
  onAgentCreate,
  onAgentUpdate,
  onAgentDelete,
  csvEnabled,
  onCsvToggle,
  uploadedFiles,
  onFilesUpload,
}) => {
  const [showForm, setShowForm] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);

  const handleCreateAgent = () => {
    setEditingAgent(null);
    setShowForm(true);
  };

  const handleEditAgent = (agent: Agent) => {
    setEditingAgent(agent);
    setShowForm(true);
  };

  const handleFormSubmit = (agentData: Omit<Agent, 'id'>) => {
    if (editingAgent) {
      onAgentUpdate(editingAgent.id, agentData);
    } else {
      onAgentCreate(agentData);
    }
    setShowForm(false);
    setEditingAgent(null);
  };

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
      </div>

      <Separator className="bg-gray-700" />

      <div className="p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">AGENTS</h3>
        <ScrollArea className="h-64">
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
                      {agent.model} • {agent.tools.length} tools
                    </div>
                  </div>
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
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      <Separator className="bg-gray-700" />

      <div className="p-4 flex-1">
        <CSVUpload
          enabled={csvEnabled}
          onToggle={onCsvToggle}
          uploadedFiles={uploadedFiles}
          onFilesUpload={onFilesUpload}
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
