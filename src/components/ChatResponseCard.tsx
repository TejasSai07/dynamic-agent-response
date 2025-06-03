
import React, { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronUp, Code, BarChart3, Brain } from 'lucide-react';
import { ChatMessage } from '@/types/Agent';

interface ChatResponseCardProps {
  message: ChatMessage;
}

export const ChatResponseCard: React.FC<ChatResponseCardProps> = ({ message }) => {
  const [codeExpanded, setCodeExpanded] = useState(false);
  const [reasoningExpanded, setReasoningExpanded] = useState(false);

  // For data analysis agent with reasoning steps
  if (message.reasoning_steps && message.reasoning_steps.length > 0) {
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
              {message.content}
            </div>
          </CardContent>
        </Card>

        {/* Reasoning Steps - Collapsible */}
        <Collapsible open={reasoningExpanded} onOpenChange={setReasoningExpanded}>
          <CollapsibleTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between bg-gray-800 border-gray-700 hover:bg-gray-700"
            >
              <div className="flex items-center space-x-2">
                <Brain className="w-4 h-4" />
                <span>Reasoning Steps ({message.reasoning_steps.length})</span>
              </div>
              {reasoningExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 space-y-3">
              {message.reasoning_steps.map((step, index) => (
                <Card key={index} className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="text-sm font-medium text-blue-400 mb-2">
                      Step {step.step_number}
                    </div>
                    <div className="text-white mb-3">{step.reasoning}</div>
                    
                    {step.code && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-400 mb-2">Code:</div>
                        <pre className="bg-gray-900 p-3 rounded text-sm text-gray-300 overflow-x-auto">
                          <code>{step.code}</code>
                        </pre>
                      </div>
                    )}
                    
                    {step.output && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-400 mb-2">Output:</div>
                        <pre className="bg-gray-900 p-3 rounded text-sm text-green-300">
                          {step.output}
                        </pre>
                      </div>
                    )}
                    
                    {step.error && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-400 mb-2">Error:</div>
                        <pre className="bg-gray-900 p-3 rounded text-sm text-red-300">
                          {step.error}
                        </pre>
                      </div>
                    )}
                    
                    {step.plot_path && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-400 mb-2">Generated Plot:</div>
                        <img
                          src={`/${step.plot_path}`}
                          alt="Generated Plot"
                          className="max-w-full h-auto rounded"
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Show plots if available */}
        {message.plot_paths && message.plot_paths.length > 0 && (
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2 mb-3">
                <BarChart3 className="w-4 h-4" />
                <span className="text-sm font-medium">Generated Plots</span>
              </div>
              <div className="space-y-3">
                {message.plot_paths.map((plotPath, index) => (
                  <img
                    key={index}
                    src={`/${plotPath}`}
                    alt={`Generated Plot ${index + 1}`}
                    className="max-w-full h-auto rounded"
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }

  // For knowledge extraction agent or simple responses
  return (
    <div className="space-y-3">
      {/* Main Response */}
      <Card className="bg-gray-800 border-gray-700">
        <CardContent className="p-4">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span className="text-sm font-medium text-green-400">Response</span>
          </div>
          <div className="text-white">
            {message.content}
          </div>
        </CardContent>
      </Card>

      {/* Tool Usage Info */}
      {message.tool_used && (
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="text-sm text-gray-400 mb-2">
              Used tool: <span className="text-blue-400">{message.tool_used}</span>
            </div>
            {message.tool_output && (
              <pre className="bg-gray-900 p-3 rounded text-sm text-gray-300 overflow-x-auto">
                {typeof message.tool_output === 'string' ? message.tool_output : JSON.stringify(message.tool_output, null, 2)}
              </pre>
            )}
          </CardContent>
        </Card>
      )}

      {/* Code Section - if available */}
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

      {/* Single Plot */}
      {message.plot_path && (
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2 mb-3">
              <BarChart3 className="w-4 h-4" />
              <span className="text-sm font-medium">Generated Plot</span>
            </div>
            <img
              src={`/${message.plot_path}`}
              alt="Generated Plot"
              className="max-w-full h-auto rounded"
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
};
