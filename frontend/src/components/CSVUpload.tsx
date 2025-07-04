import React, { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Upload, X, FileText } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

interface CSVUploadProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  uploadedFiles: File[];
  onFilesUpload: (files: File[]) => void;
}

export const CSVUpload: React.FC<CSVUploadProps> = ({
  enabled,
  onToggle,
  uploadedFiles = [], // Add default value here
  onFilesUpload,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const csvFiles = files.filter(file => 
      file.name.endsWith('.csv') || 
      file.name.endsWith('.xlsx') || 
      file.name.endsWith('.xls')
    );
    
    if (csvFiles.length > 0 && typeof onFilesUpload === 'function') {
      // Add new files to existing files
      const newFiles = [...uploadedFiles, ...csvFiles];
      onFilesUpload(newFiles);
      
      console.log('CSV files added:', csvFiles.map(f => f.name));
    }
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index: number) => {
    if (typeof onFilesUpload === 'function') {
      const newFiles = uploadedFiles.filter((_, i) => i !== index);
      onFilesUpload(newFiles);
    }
  };

  const clearAllFiles = () => {
    if (typeof onFilesUpload === 'function') {
      onFilesUpload([]);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2">
        <Switch
          id="csv-toggle"
          checked={enabled}
          onCheckedChange={onToggle}
        />
        <Label htmlFor="csv-toggle" className="text-sm font-medium">
          Enable CSV Upload
        </Label>
      </div>

      {enabled && (
        <div className="space-y-3">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".csv,.xlsx,.xls"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          <Button
            onClick={() => fileInputRef.current?.click()}
            variant="outline"
            className="w-full bg-gray-700 border-gray-600 hover:bg-gray-600"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload CSV Files
          </Button>

          {uploadedFiles.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-gray-400">
                  Uploaded Files ({uploadedFiles.length}):
                </Label>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={clearAllFiles}
                  className="h-6 text-xs text-gray-400 hover:text-red-400"
                >
                  Clear All
                </Button>
              </div>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {uploadedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-700 rounded px-3 py-2"
                  >
                    <div className="flex items-center space-x-2 flex-1 min-w-0">
                      <FileText className="w-4 h-4 text-green-400 flex-shrink-0" />
                      <span className="text-sm truncate" title={file.name}>
                        {file.name}
                      </span>
                      <span className="text-xs text-gray-400 flex-shrink-0">
                        ({(file.size / 1024).toFixed(1)}KB)
                      </span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => removeFile(index)}
                      className="h-6 w-6 p-0 text-gray-400 hover:text-red-400 flex-shrink-0"
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};