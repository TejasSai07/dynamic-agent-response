import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight, Code, BarChart3, Brain, CheckCircle, Star, Play } from 'lucide-react';
import { ChatMessage } from '@/types/Agent';

interface ChatResponseCardProps {
  message: ChatMessage;
}

export const ChatResponseCard: React.FC<ChatResponseCardProps> = ({ message }) => {
  const [showCode, setShowCode] = useState(false);

  // Helper to parse markdown-style bold (**text**) to <strong>
  const formatContent = (content: string) => {
    return content.split(/(\*\*.*?\*\*)/).map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} className="font-bold text-white">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  const isFinalAnswer = message.isComplete === true && message.finalAnswer;

  // 👇 FIX: Combine possible sources of plot paths
  const plotPaths =
    message.plot_paths ||
    message.final_output?.plot_paths ||
    [];

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">

      {/* ✅ Final Answer */}
      {isFinalAnswer && (
        <div className="bg-gradient-to-r from-green-900/50 to-blue-900/50 border border-green-500/30 rounded-lg p-4 mb-4">
          <div className="flex items-center mb-2">
            <Star className="w-5 h-5 text-yellow-400 mr-2" />
            <h3 className="text-lg font-bold text-white">Final Answer</h3>
            <CheckCircle className="w-5 h-5 text-green-400 ml-2" />
          </div>
          <div className="text-green-100 whitespace-pre-wrap">
            {formatContent(message.finalAnswer)}
          </div>
        </div>
      )}

      {/* Reasoning (non-final) */}
      {!isFinalAnswer && (
        <div className="border-l-4 border-blue-500 pl-4">
          <div className="flex items-center mb-2">
            <Brain className="w-4 h-4 mr-2 text-blue-400" />
            <span className="text-sm font-medium text-blue-300">Reasoning Step</span>
          </div>
          <div className="text-white whitespace-pre-wrap">
            {formatContent(message.content)}
          </div>
        </div>
      )}

      {/* Executed Code */}
      {message.code && (
        <Collapsible open={showCode} onOpenChange={setShowCode}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-green-400 hover:text-green-300"
            >
              {showCode ? <ChevronDown className="w-4 h-4 mr-2" /> : <ChevronRight className="w-4 h-4 mr-2" />}
              <Code className="w-4 h-4 mr-2" />
              {isFinalAnswer ? 'View Final Code' : 'View Code Executed'}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="bg-gray-900 rounded p-3 mt-2">
              <pre className="text-sm text-green-400 overflow-x-auto">
                <code>{message.code}</code>
              </pre>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Execution Output */}
      {message.output && (
        <div className="bg-gray-900 rounded p-3 border-l-4 border-yellow-500">
          <div className="flex items-center mb-2">
            <Play className="w-4 h-4 mr-2 text-yellow-400" />
            <span className="text-sm font-medium text-yellow-300">Execution Output</span>
          </div>
          <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
            {message.output}
          </pre>
        </div>
      )}

      {/* Execution Error */}
      {message.error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded p-3">
          <div className="flex items-center mb-2">
            <span className="text-sm font-medium text-red-300">Error</span>
          </div>
          <pre className="text-sm text-red-400 overflow-x-auto whitespace-pre-wrap">
            {message.error}
          </pre>
        </div>
      )}

      {/* ✅ Plot Paths (final or intermediate) */}
      {plotPaths.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center text-green-400">
            <BarChart3 className="w-4 h-4 mr-2" />
            <span className="text-sm font-medium">Generated Plots ({plotPaths.length})</span>
          </div>
          <div className="grid grid-cols-1 gap-4">
            {plotPaths.map((plotPath, index) => (
              <div key={`${message.timestamp}-${index}`} className="bg-gray-900 rounded-lg p-3 border border-gray-700">
                <img
                  src={`${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/${plotPath.replace(/^\/?/, '')}`}
                  alt={`Plot ${index + 1}`}
                  className="w-full h-auto rounded border border-gray-600 max-w-full"
                  style={{ maxHeight: '500px', objectFit: 'contain' }}
                  onError={(e) => {
                    console.error('Failed to load plot:', plotPath);
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Step-by-step Analysis Reasoning */}
      {message.reasoning_steps && message.reasoning_steps.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center text-purple-400">
            <Brain className="w-4 h-4 mr-2" />
            <span className="text-sm font-medium">Analysis Steps ({message.reasoning_steps.length})</span>
          </div>
          {message.reasoning_steps.map((step, index) => (
            <div key={index} className="bg-gray-900 rounded p-3 border-l-4 border-purple-500">
              <div className="flex items-center mb-2">
                <span className="bg-purple-600 text-white text-xs px-2 py-1 rounded mr-2">
                  Step {step.step_number}
                </span>
                {step.step_number === message.reasoning_steps!.length && (
                  <CheckCircle className="w-4 h-4 text-green-400" />
                )}
              </div>

              {step.reasoning && (
                <div className="mb-3">
                  <h4 className="text-sm font-medium text-purple-300 mb-1">Reasoning:</h4>
                  <p className="text-sm text-gray-300 leading-relaxed">{step.reasoning}</p>
                </div>
              )}

              {step.next_step && (
                <div className="mb-3">
                  <h4 className="text-sm font-medium text-blue-300 mb-1">Next Step:</h4>
                  <p className="text-sm text-gray-300 leading-relaxed">{step.next_step}</p>
                </div>
              )}

              {step.code && (
                <div className="mb-3">
                  <h4 className="text-sm font-medium text-green-300 mb-1">Code Executed:</h4>
                  <pre className="text-xs text-green-400 bg-gray-800 p-2 rounded overflow-x-auto">
                    <code>{step.code}</code>
                  </pre>
                </div>
              )}

              {step.output && (
                <div className="mb-2">
                  <h4 className="text-sm font-medium text-blue-300 mb-1">Output:</h4>
                  <pre className="text-xs text-gray-300 bg-gray-800 p-2 rounded overflow-x-auto max-h-32">
                    {step.output}
                  </pre>
                </div>
              )}

              {step.error && (
                <div className="mb-2">
                  <h4 className="text-sm font-medium text-red-300 mb-1">Error:</h4>
                  <pre className="text-xs text-red-400 bg-gray-800 p-2 rounded overflow-x-auto">
                    {step.error}
                  </pre>
                </div>
              )}

              {/* Step-specific plot fallback */}
              {step.plot_path && (
                <div className="mt-2">
                  <h4 className="text-sm font-medium text-green-300 mb-1">Generated Plot:</h4>
                  <div className="bg-gray-800 rounded p-2">
                    <img
                      src={`${import.meta.env.VITE_API_URL}/${step.plot_path}`}
                      alt={`Step ${step.step_number} Plot`}
                      className="w-full h-auto rounded max-w-md"
                      onError={(e) => {
                        console.error('Failed to load plot:', step.plot_path);
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
