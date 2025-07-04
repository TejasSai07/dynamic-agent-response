
import React, { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Upload, X, FileText } from 'lucide-react';

interface FileUploadProps {
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  uploadedFile,
  onFileUpload,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const removeFile = () => {
    onFileUpload(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-4">
      <Label className="text-sm font-medium">File Upload</Label>
      
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.txt,.json,.xlsx,.xls"
        onChange={handleFileUpload}
        className="hidden"
      />
      
      <Button
        onClick={() => fileInputRef.current?.click()}
        variant="outline"
        className="w-full bg-gray-700 border-gray-600 hover:bg-gray-600"
      >
        <Upload className="w-4 h-4 mr-2" />
        Upload File
      </Button>

      {uploadedFile && (
        <div className="space-y-2">
          <Label className="text-xs text-gray-400">Uploaded File:</Label>
          <div className="flex items-center justify-between bg-gray-700 rounded px-3 py-2">
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <FileText className="w-4 h-4 text-blue-400 flex-shrink-0" />
              <span className="text-sm truncate" title={uploadedFile.name}>
                {uploadedFile.name}
              </span>
              <span className="text-xs text-gray-400 flex-shrink-0">
                ({(uploadedFile.size / 1024).toFixed(1)}KB)
              </span>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={removeFile}
              className="h-6 w-6 p-0 text-gray-400 hover:text-red-400 flex-shrink-0"
            >
              <X className="w-3 h-3" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
