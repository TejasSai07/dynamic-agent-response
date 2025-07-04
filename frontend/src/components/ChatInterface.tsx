import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Bot, User, Upload, X, FileText } from 'lucide-react';
import { Agent, ChatMessage } from '@/types/Agent';
import { ChatResponseCard } from './ChatResponseCard';
import { useToast } from '@/hooks/use-toast';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

interface ChatInterfaceProps {
  selectedAgent: Agent | null;
  conversationId: string | null;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
  csvFiles: File[];
  onCsvFilesUpload: (files: File[]) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  selectedAgent,
  conversationId,
  uploadedFile,
  onFileUpload,
  csvFiles = [],
  onCsvFilesUpload,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationFile, setConversationFile] = useState<File | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    if (conversationId) {
      fetchMessages();
    } else {
      setMessages([]);
    }
  }, [conversationId]);
  
  const fetchMessages = async () => {
    if (!conversationId) return;
    try {
      const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      const processedMessages: ChatMessage[] = data.map((msg: any) => {
        if (msg.role === 'assistant' && typeof msg.content === 'string') {
          try {
            if (msg.content.trim().startsWith('{')) {
              const parsed = JSON.parse(msg.content);
              
              // For custom agents, check if this is truly complete
              const isActuallyComplete = parsed.is_complete === true;
              
              return {
                ...msg,
                content: parsed.updated_answer || parsed.reasoning || msg.content,
                code: parsed.code ?? undefined,
                finalAnswer: isActuallyComplete ? (parsed.updated_answer || parsed.reasoning) : undefined,
                isComplete: isActuallyComplete,
                plot_paths: parsed.plot_paths ?? [],
                output: parsed.output || undefined,
                error: parsed.error || undefined,
              };
            }
          } catch (e) {
            console.log('Content is not JSON, treating as plain text:', e);
          }
        }
        return msg;
      });

      setMessages(processedMessages);
    } catch (error) {
      console.error('Error fetching messages:', error);
      toast({ title: 'Error', description: 'Failed to fetch messages.', variant: 'destructive' });
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setConversationFile(file);
  };

  const removeConversationFile = () => {
    setConversationFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const uploadFiles = async (files: File[]) => {
    const uploadedPaths: string[] = [];
    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData });
        if (response.ok) {
          const data = await response.json();
          uploadedPaths.push(data.file_path);
        }
      } catch (error) {
        console.error('Upload failed for file:', file.name, error);
      }
    }
    return uploadedPaths;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !conversationId || isLoading) return;
    const userMessage: ChatMessage = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      const filesToUpload = [conversationFile, uploadedFile, ...csvFiles].filter(Boolean);
      const file_paths = filesToUpload.length > 0 ? await uploadFiles(filesToUpload) : [];
      const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: currentInput, file_paths }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchMessages();
      toast({ title: 'Message sent', description: 'Your message has been sent successfully!' });
    } catch (error) {
      console.error('Error sending message:', error);
      toast({ title: 'Error', description: 'Failed to send message.', variant: 'destructive' });
      setInputValue(currentInput);
    } finally {
      setIsLoading(false);
    }
  };

  if (!selectedAgent) return (
    <div className="flex-1 flex items-center justify-center bg-gray-900">
      <div className="text-center text-gray-400">
        <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <h2 className="text-xl font-medium mb-2">No Agent Selected</h2>
        <p>Select an agent from the sidebar to start chatting</p>
      </div>
    </div>
  );

  if (!conversationId) return (
    <div className="flex-1 flex items-center justify-center bg-gray-900">
      <div className="text-center text-gray-400">
        <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <h2 className="text-xl font-medium mb-2">No Conversation Selected</h2>
        <p>Start a new conversation to begin chatting with {selectedAgent.name}</p>
      </div>
    </div>
  );

  const totalFileCount = [conversationFile, uploadedFile, ...csvFiles].filter(Boolean).length;

  return (
    <div className="flex-1 flex flex-col bg-gray-900">
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">{selectedAgent?.name}</h2>
            <p className="text-sm text-gray-400">
              {selectedAgent?.model_type} • Memory: {selectedAgent?.memory_enabled ? 'On' : 'Off'}
              {selectedAgent?.csv_upload_enabled && ' • CSV Upload Enabled'}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.txt,.json,.xlsx"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button onClick={() => fileInputRef.current?.click()} variant="outline" size="sm" className="bg-gray-700 border-gray-600 hover:bg-gray-600">
              <Upload className="w-4 h-4 mr-2" /> Add File
            </Button>
            {totalFileCount > 0 && (
              <div className="text-sm text-gray-400 flex items-center">
                <FileText className="w-4 h-4 mr-1" /> Files: {totalFileCount}
                {conversationFile && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={removeConversationFile}
                    className="h-4 w-4 p-0 ml-2 text-gray-400 hover:text-red-400"
                  >
                    <X className="w-3 h-3" />
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
        <div className="space-y-4 max-w-5xl mx-auto">
          {messages.map((message, index) => (
            <div key={`${message.timestamp}-${index}`}>
              {message.role === 'user' ? (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4" />
                  </div>
                  <div className="flex-1 bg-gray-800 rounded-lg p-4">
                    <p className="text-white whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              ) : message.role === 'assistant' ? (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="flex-1">
                    <ChatResponseCard message={message} />
                  </div>
                </div>
              ) : null}
            </div>
          ))}
          {isLoading && (
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4" />
              </div>
              <div className="flex-1 bg-gray-800 rounded-lg p-4">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-400"></div>
                  <p className="text-gray-400">Thinking...</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="border-t border-gray-700 p-4">
        <div className="max-w-5xl mx-auto">
          <div className="flex space-x-2">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder={`Message ${selectedAgent?.name}...`}
              className="flex-1 bg-gray-800 border-gray-600 text-white resize-none"
              rows={1}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
