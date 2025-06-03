import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Agent, AgentDefinition } from '@/types/Agent';

interface AgentFormProps {
  agent: Agent | null;
  onSubmit: (agent: Omit<AgentDefinition, 'id'>) => void;
  onCancel: () => void;
}

const AVAILABLE_TOOLS = [
  'current_date',
  'web_search',
  'faiss_knn_search',
  'extract_key_terms',
  'knowledge_graph',
];

const AVAILABLE_MODELS = [
  'gpt-4o',
  'gpt-4o-mini',
  'gpt-4-turbo',
  'gpt-3.5-turbo',
];

export const AgentForm: React.FC<AgentFormProps> = ({
  agent,
  onSubmit,
  onCancel,
}) => {
  const [formData, setFormData] = useState({
    name: '',
    model_type: 'gpt-4o',
    system_prompt: '',
    tools: [] as string[],
    memory_enabled: false,
    tasks: [] as string[],
  });

  useEffect(() => {
    if (agent) {
      setFormData({
        name: agent.name,
        model_type: agent.model_type,
        system_prompt: '', // This would need to be fetched from the backend
        tools: [], // This would need to be fetched from the backend
        memory_enabled: agent.memory_enabled || false,
        tasks: [],
      });
    }
  }, [agent]);

  const handleToolToggle = (tool: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      tools: checked
        ? [...prev.tools, tool]
        : prev.tools.filter(t => t !== tool),
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <Dialog open={true} onOpenChange={() => onCancel()}>
      <DialogContent className="sm:max-w-[500px] bg-gray-800 text-white border-gray-700">
        <DialogHeader>
          <DialogTitle>
            {agent ? 'Edit Agent' : 'Create New Agent'}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name">Agent Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              className="bg-gray-700 border-gray-600 text-white"
              required
            />
          </div>

          <div>
            <Label htmlFor="model">Model</Label>
            <Select value={formData.model_type} onValueChange={(value) => setFormData(prev => ({ ...prev, model_type: value }))}>
              <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-700 border-gray-600">
                {AVAILABLE_MODELS.map((model) => (
                  <SelectItem key={model} value={model} className="text-white hover:bg-gray-600">
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="prompt">System Prompt</Label>
            <Textarea
              id="prompt"
              value={formData.system_prompt}
              onChange={(e) => setFormData(prev => ({ ...prev, system_prompt: e.target.value }))}
              className="bg-gray-700 border-gray-600 text-white min-h-[100px]"
              placeholder="You are a helpful assistant..."
              required
            />
          </div>

          <div>
            <Label>Tools</Label>
            <div className="space-y-2 mt-2">
              {AVAILABLE_TOOLS.map((tool) => (
                <div key={tool} className="flex items-center space-x-2">
                  <Checkbox
                    id={tool}
                    checked={formData.tools.includes(tool)}
                    onCheckedChange={(checked) => handleToolToggle(tool, checked as boolean)}
                    className="border-gray-600"
                  />
                  <Label htmlFor={tool} className="text-sm">
                    {tool.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="memory"
              checked={formData.memory_enabled}
              onCheckedChange={(checked) => setFormData(prev => ({ ...prev, memory_enabled: checked }))}
            />
            <Label htmlFor="memory">Enable Memory</Label>
          </div>

          <div className="flex gap-2 pt-4">
            <Button type="button" variant="outline" onClick={onCancel} className="flex-1">
              Cancel
            </Button>
            <Button type="submit" className="flex-1 bg-blue-600 hover:bg-blue-700">
              {agent ? 'Update' : 'Create'} Agent
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};