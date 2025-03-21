import React, { useState, useEffect, useRef } from 'react';
import { Send, UploadCloud, Book, Loader2, RefreshCcw, Clock, X, AlertCircle, ChevronUp, ChevronDown, FileText, Search, Menu, MessageSquare } from 'lucide-react';

// Define TypeScript interfaces
interface Machine {
  id: string;
  name: string;
  model: string;
  description?: string;
}

interface QueryHistory {
  query: string;
  response: string;
  machine: Machine;
  timestamp: Date;
}

const ServiceManualRAG = () => {
  // State management
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [machines, setMachines] = useState<Machine[]>([]);
  const [selectedMachine, setSelectedMachine] = useState<Machine | null>(null);
  const [machinesLoading, setMachinesLoading] = useState(true);
  const [error, setError] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState('');
  const [showUpload, setShowUpload] = useState(false);
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set());
  
  const responseRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // API endpoint (would be configured properly in a real app)
  const API_URL = 'http://localhost:8000';
  
  // Load from local storage on component mount
  useEffect(() => {
    fetchMachines();
    
    // Load history from localStorage
    const savedHistory = localStorage.getItem('queryHistory');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        setQueryHistory(parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        })));
      } catch (e) {
        console.error('Failed to parse saved query history');
      }
    }
    
    // Load selected machine from localStorage
    const savedMachine = localStorage.getItem('selectedMachine');
    if (savedMachine) {
      try {
        setSelectedMachine(JSON.parse(savedMachine));
      } catch (e) {
        console.error('Failed to parse saved machine');
      }
    }
  }, []);
  
  // Scroll to bottom of response when it changes
  useEffect(() => {
    if (responseRef.current && response) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);
  
  // Save history to localStorage when it changes
  useEffect(() => {
    if (queryHistory.length > 0) {
      localStorage.setItem('queryHistory', JSON.stringify(queryHistory));
    }
  }, [queryHistory]);
  
  // Save selected machine to localStorage
  useEffect(() => {
    if (selectedMachine) {
      localStorage.setItem('selectedMachine', JSON.stringify(selectedMachine));
    }
  }, [selectedMachine]);
  
  // Fetch machines from API
  const fetchMachines = async () => {
    setMachinesLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/machines`);
      if (!response.ok) {
        throw new Error('Failed to fetch machines');
      }
      
      const data = await response.json();
      setMachines(data.machines || []);
    } catch (err) {
      console.error('Error fetching machines:', err);
      setError('Failed to load machines. Please refresh and try again.');
    } finally {
      setMachinesLoading(false);
    }
  };
  
  // Handle query submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) return;
    if (!selectedMachine) {
      setError('Please select a machine first');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          machine_id: selectedMachine.id,
          top_k: 10
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch response');
      }
      
      const data = await response.json();
      setResponse(data.response);
      
      // Add to history
      const newHistoryItem: QueryHistory = {
        query: query.trim(),
        response: data.response,
        machine: selectedMachine,
        timestamp: new Date()
      };
      
      setQueryHistory(prev => [newHistoryItem, ...prev].slice(0, 50)); // Keep last 50 queries
      
    } catch (err) {
      console.error('Error querying API:', err);
      setError('Failed to get a response. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle machine selection change
  const handleMachineChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const machineId = e.target.value;
    if (!machineId) {
      setSelectedMachine(null);
      return;
    }
    
    const machine = machines.find(m => m.id === machineId);
    setSelectedMachine(machine || null);
    setResponse(''); // Clear previous responses when changing machines
  };
  
  // Handle document upload
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!fileInputRef.current?.files?.length) {
      setUploadError('Please select a file to upload');
      return;
    }
    
    if (!selectedMachine) {
      setUploadError('Please select a machine first');
      return;
    }
    
    const file = fileInputRef.current.files[0];
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF files are supported');
      return;
    }
    
    setUploading(true);
    setUploadError('');
    setUploadSuccess('');
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('machine_id', selectedMachine.id);
    
    try {
      const response = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }
      
      const data = await response.json();
      setUploadSuccess(`Successfully uploaded ${file.name}. Processing ${data.chunk_count} chunks.`);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err) {
      console.error('Error uploading document:', err);
      setUploadError(`Failed to upload document: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };
  
  // Load a previous query from history
  const loadFromHistory = (item: QueryHistory) => {
    // If we need to switch machines
    if (!selectedMachine || selectedMachine.id !== item.machine.id) {
      setSelectedMachine(item.machine);
    }
    
    setQuery(item.query);
    setResponse(item.response);
    setShowHistory(false);
  };
  
  // Clear query history
  const clearHistory = () => {
    setQueryHistory([]);
    localStorage.removeItem('queryHistory');
    setShowHistory(false);
  };
  
  // Toggle citation expansion
  const toggleCitation = (citation: string) => {
    setExpandedCitations(prev => {
      const newSet = new Set(prev);
      if (newSet.has(citation)) {
        newSet.delete(citation);
      } else {
        newSet.add(citation);
      }
      return newSet;
    });
  };
  
  // Format the response with proper styling for citations
  const formatResponse = (text: string) => {
    if (!text) return null;
    
    // Extract all citations with their surrounding text
    const citationRegex = /(\[Document: [^\]]+\])/g;
    const parts = text.split(citationRegex);
    
    return (
      <div className="whitespace-pre-line">
        {parts.map((part, index) => {
          if (part.match(citationRegex)) {
            // This is a citation
            return (
              <div key={index} className="inline-block group relative">
                <span 
                  className="text-blue-600 font-medium cursor-pointer hover:underline"
                  onClick={() => toggleCitation(part)}
                >
                  {part}
                  {expandedCitations.has(part) ? 
                    <ChevronUp className="inline-block ml-1 w-4 h-4" /> : 
                    <ChevronDown className="inline-block ml-1 w-4 h-4" />
                  }
                </span>
                {expandedCitations.has(part) && (
                  <div className="mt-2 mb-2 p-3 bg-blue-50 rounded-md border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-1">{part}</h4>
                    <p className="text-sm text-gray-700">
                      This citation references the exact location in the service manual 
                      where this information was found.
                    </p>
                  </div>
                )}
              </div>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </div>
    );
  };
  
  // Format timestamp for history display
  const formatTimestamp = (date: Date) => {
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  // Detect if query contains an error code and show a hint
  const hasErrorCode = (text: string) => {
    return /(\d+-\d+|\d+[A-Za-z]\d+|[A-Za-z]\d+)/.test(text);
  };

  // Truncate text to a specific length
  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} bg-white border-r border-gray-200 transition-all duration-300 overflow-hidden flex flex-col`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Book className="text-blue-600" size={20} />
            <h1 className="font-bold text-lg text-gray-800">Service Manual</h1>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="text-gray-500 hover:text-gray-700">
            <X size={18} />
          </button>
        </div>
        
        {/* Machine Selection */}
        <div className="p-4 border-b border-gray-200">
          <label className="block text-sm font-medium text-gray-700 mb-2">Machine Selection</label>
          <div className="flex space-x-2">
            <select
              className="flex-1 px-3 py-2 rounded-md border border-gray-300 focus:ring-blue-500 focus:border-blue-500"
              onChange={handleMachineChange}
              value={selectedMachine?.id || ''}
              disabled={machinesLoading}
            >
              <option value="">Select a machine...</option>
              {machines.map(machine => (
                <option key={machine.id} value={machine.id}>
                  {machine.name} ({machine.model})
                </option>
              ))}
            </select>
            <button 
              onClick={fetchMachines} 
              className="p-2 bg-gray-100 rounded-md hover:bg-gray-200 text-gray-700"
              disabled={machinesLoading}
              title="Refresh machines list"
            >
              {machinesLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <RefreshCcw className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
        
        {/* Machine Info */}
        {selectedMachine && (
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-md font-semibold text-gray-800">
              {selectedMachine.name} ({selectedMachine.model})
            </h2>
            {selectedMachine.description && (
              <p className="text-sm text-gray-600 mt-1">{selectedMachine.description}</p>
            )}
          </div>
        )}
        
        {/* Manual Upload Section */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={() => setShowUpload(!showUpload)}
            className="flex items-center space-x-2 text-blue-600 hover:text-blue-800"
          >
            <UploadCloud size={16} />
            <span>{showUpload ? 'Hide Upload Form' : 'Upload Service Manual'}</span>
          </button>
          
          {showUpload && (
            <form onSubmit={handleUpload} className="mt-3">
              <div className="mb-3">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".pdf"
                  className="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100"
                />
              </div>
              
              {uploadError && (
                <div className="mb-3 p-2 bg-red-50 text-red-700 text-sm rounded-md flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{uploadError}</span>
                </div>
              )}
              
              {uploadSuccess && (
                <div className="mb-3 p-2 bg-green-50 text-green-700 text-sm rounded-md">
                  {uploadSuccess}
                </div>
              )}
              
              <button
                type="submit"
                disabled={uploading || !selectedMachine}
                className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md 
                  disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {uploading ? (
                  <>
                    <Loader2 className="animate-spin mr-2 w-4 h-4" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <UploadCloud className="mr-2 w-4 h-4" />
                    <span>Upload PDF</span>
                  </>
                )}
              </button>
            </form>
          )}
        </div>
        
        {/* Query History Section */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="flex items-center space-x-2 text-blue-600 hover:text-blue-800"
          >
            <Clock size={16} />
            <span>{showHistory ? 'Hide Query History' : 'Show Query History'}</span>
          </button>
        </div>
        
        {/* Query History List */}
        {showHistory && (
          <div className="flex-1 overflow-auto p-2">
            {queryHistory.length === 0 ? (
              <p className="text-gray-500 text-sm p-2">No query history yet.</p>
            ) : (
              <>
                <div className="flex justify-end mb-2">
                  <button 
                    onClick={clearHistory}
                    className="text-xs text-red-600 hover:text-red-800"
                  >
                    Clear History
                  </button>
                </div>
                <div className="space-y-2">
                  {queryHistory.map((item, index) => (
                    <div 
                      key={index} 
                      className="p-3 bg-white border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer"
                      onClick={() => loadFromHistory(item)}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-medium text-gray-800">{truncateText(item.query, 40)}</span>
                        <span className="text-xs text-gray-500">{formatTimestamp(item.timestamp)}</span>
                      </div>
                      <div className="text-xs text-gray-600 truncate">
                        {item.machine.name} ({item.machine.model})
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <header className="bg-blue-600 text-white p-4 shadow-md">
          <div className="container mx-auto flex items-center">
            {!sidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} className="mr-3 text-white">
                <Menu size={24} />
              </button>
            )}
            <div className="flex items-center space-x-2">
              <Book size={24} />
              <h1 className="text-xl font-bold">Service Manual Assistant</h1>
            </div>
            {selectedMachine && (
              <div className="ml-auto text-sm bg-blue-700 px-3 py-1 rounded-full">
                {selectedMachine.name} ({selectedMachine.model})
              </div>
            )}
          </div>
        </header>
        
        {/* Main Content Area */}
        <div className="flex-1 container mx-auto p-4 flex flex-col overflow-hidden">
          {/* Error Message */}
          {error && (
            <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4 border border-red-100 flex items-center space-x-2">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}
          
          {/* Response Area */}
          <div ref={responseRef} className="flex-1 bg-white rounded-lg shadow-md p-6 mb-4 overflow-y-auto">
            {response ? (
              <div className="prose max-w-none">
                {formatResponse(response)}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-gray-400">
                {selectedMachine ? (
                  <>
                    <FileText size={48} className="mb-4 text-gray-300" />
                    <p>Ask a question about the {selectedMachine.name} service manual</p>
                    <p className="text-sm mt-2">Try asking about maintenance procedures, error codes, or specific parts</p>
                  </>
                ) : (
                  <>
                    <Book size={48} className="mb-4 text-gray-300" />
                    <p>Please select a machine from the sidebar to begin</p>
                  </>
                )}
              </div>
            )}
            
            {loading && (
              <div className="absolute inset-0 bg-white bg-opacity-60 flex items-center justify-center">
                <div className="bg-white p-4 rounded-lg shadow-md flex items-center space-x-3">
                  <Loader2 size={30} className="animate-spin text-blue-600" />
                  <div>
                    <p className="font-medium text-gray-800">Searching manual...</p>
                    <p className="text-sm text-gray-500">Finding the most relevant information</p>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Query Input */}
          <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-3 flex space-x-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about the service manual..."
                className="w-full px-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
                disabled={loading || !selectedMachine}
              />
              {hasErrorCode(query) && (
                <div className="absolute right-3 top-3 text-blue-600">
                  <AlertCircle size={18} title="Error code detected" />
                </div>
              )}
            </div>
            <button
              type="submit"
              className="bg-blue-600 text-white px-4 py-3 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-400 flex items-center space-x-2"
              disabled={loading || !query.trim() || !selectedMachine}
            >
              {loading ? (
                <Loader2 size={20} className="animate-spin" />
              ) : (
                <>
                  <span>Ask</span>
                  <Send size={16} />
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ServiceManualRAG;
