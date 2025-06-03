
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronUp, Code, BarChart3 } from 'lucide-react';
import { ChatMessage } from '@/types/Agent';

interface ChatResponseCardProps {
  message: ChatMessage;
}

export const ChatResponseCard: React.FC<ChatResponseCardProps> = ({ message }) => {
  const [codeExpanded, setCodeExpanded] = useState(false);
  const [graphExpanded, setGraphExpanded] = useState(false);

  return (
    <div className="space-y-3">
      {/* Final Answer - Always Visible */}
      <Card className="bg-gray-800 border-gray-700">
        <CardContent className="p-4">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span className="text-sm font-medium text-green-400">Final Answer</span>
          </div>
          <div className="text-white font-medium">
            {message.finalAnswer}
          </div>
        </CardContent>
      </Card>

      {/* Code Section - Collapsible */}
      {message.code && (
        <Collapsible open={codeExpanded} onOpenChange={setCodeExpanded}>
          <CollapsibleTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between bg-gray-800 border-gray-700 hover:bg-gray-700"
            >
              <div className="flex items-center space-x-2">
                <Code className="w-4 h-4" />
                <span>Generated Code</span>
              </div>
              {codeExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <Card className="mt-2 bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <pre className="bg-gray-900 p-3 rounded text-sm text-gray-300 overflow-x-auto">
                  <code>{message.code}</code>
                </pre>
              </CardContent>
            </Card>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Graph Section - Collapsible */}
      {message.graph && (
        <Collapsible open={graphExpanded} onOpenChange={setGraphExpanded}>
          <CollapsibleTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between bg-gray-800 border-gray-700 hover:bg-gray-700"
            >
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span>Generated Graph</span>
              </div>
              {graphExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <Card className="mt-2 bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <img
                  src={`data:image/png;base64,${message.graph}`}
                  alt="Generated Graph"
                  className="max-w-full h-auto rounded"
                />
              </CardContent>
            </Card>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
};
