
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { X } from 'lucide-react';
import { Agent, AgentDefinition } from '@/types/Agent';

interface AgentFormProps {
  agent: Agent | null;
  onSubmit: (agent: Omit<AgentDefinition, 'id'>) => void;
  onCancel: () => void;
}

export const AgentForm: React.FC<AgentFormProps> = ({ agent, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    system_prompt: '',
    model_type: 'gpt-4o-mini',
    memory_enabled: true,
    csv_upload_enabled: false,
  });

  useEffect(() => {
    if (agent) {
      setFormData({
        name: agent.name,
        description: agent.description,
        system_prompt: agent.system_prompt,
        model_type: agent.model_type,
        memory_enabled: agent.memory_enabled,
        csv_upload_enabled: agent.csv_upload_enabled || false,
      });
    }
  }, [agent]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      ...formData,
      is_built_in: false,
    });
  };

  const handleChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold">
            {agent ? 'Edit Agent' : 'Create New Agent'}
          </h2>
          <Button
            onClick={onCancel}
            variant="ghost"
            size="sm"
            className="text-gray-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name">Agent Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              className="bg-gray-700 border-gray-600 text-white"
              required
            />
          </div>

          <div>
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              className="bg-gray-700 border-gray-600 text-white"
              rows={3}
            />
          </div>

          <div>
            <Label htmlFor="system_prompt">System Prompt</Label>
            <Textarea
              id="system_prompt"
              value={formData.system_prompt}
              onChange={(e) => handleChange('system_prompt', e.target.value)}
              className="bg-gray-700 border-gray-600 text-white"
              rows={6}
              placeholder="Define the agent's role, personality, and instructions..."
              required
            />
          </div>

          <div>
            <Label htmlFor="model_type">Model Type</Label>
            <Select value={formData.model_type} onValueChange={(value) => handleChange('model_type', value)}>
              <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gpt-4o-mini">GPT-4O Mini</SelectItem>
                <SelectItem value="gpt-4o">GPT-4O</SelectItem>
                <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="memory_enabled"
              checked={formData.memory_enabled}
              onCheckedChange={(checked) => handleChange('memory_enabled', checked)}
            />
            <Label htmlFor="memory_enabled">Enable Memory</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="csv_upload_enabled"
              checked={formData.csv_upload_enabled}
              onCheckedChange={(checked) => handleChange('csv_upload_enabled', checked)}
            />
            <Label htmlFor="csv_upload_enabled">Enable CSV Upload</Label>
          </div>

          <div className="flex space-x-3 pt-4">
            <Button type="submit" className="bg-blue-600 hover:bg-blue-700">
              {agent ? 'Update Agent' : 'Create Agent'}
            </Button>
            <Button type="button" onClick={onCancel} variant="outline">
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};