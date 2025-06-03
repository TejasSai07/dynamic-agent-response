
import React, { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Upload, X, FileText } from 'lucide-react';

interface CSVUploadProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  uploadedFiles: File[];
  onFilesUpload: (files: File[]) => void;
}

export const CSVUpload: React.FC<CSVUploadProps> = ({
  enabled,
  onToggle,
  uploadedFiles,
  onFilesUpload,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const csvFiles = files.filter(file => file.name.endsWith('.csv'));
    
    if (csvFiles.length > 0) {
      // Upload files to backend
      const formData = new FormData();
      csvFiles.forEach(file => formData.append('files', file));
      
      try {
        const response = await fetch('/upload_csv', {
          method: 'POST',
          body: formData,
        });
        
        if (response.ok) {
          onFilesUpload([...uploadedFiles, ...csvFiles]);
        }
      } catch (error) {
        console.error('Error uploading CSV files:', error);
      }
    }
  };

  const removeFile = (index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index);
    onFilesUpload(newFiles);
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
            accept=".csv"
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
              <Label className="text-xs text-gray-400">Uploaded Files:</Label>
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between bg-gray-700 rounded px-3 py-2"
                >
                  <div className="flex items-center space-x-2">
                    <FileText className="w-4 h-4 text-blue-400" />
                    <span className="text-sm truncate">{file.name}</span>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => removeFile(index)}
                    className="h-6 w-6 p-0 text-gray-400 hover:text-red-400"
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
