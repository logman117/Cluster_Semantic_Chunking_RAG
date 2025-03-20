import React, { useState, useEffect, useRef } from 'react';
import { Send, Search, Book, Loader2, RefreshCcw } from 'lucide-react';

const ServiceManualRAG = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [machinesLoading, setMachinesLoading] = useState(true);
  const [error, setError] = useState('');
  const responseRef = useRef(null);
  
  // API endpoint (would be configured properly in a real app)
  const API_URL = 'http://localhost:8000';
  
  // Fetch machines on component mount
  useEffect(() => {
    fetchMachines();
  }, []);
  
  // Scroll to bottom of response when it changes
  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);
  
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
  
  const handleSubmit = async (e) => {
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
          top_k: 5
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch response');
      }
      
      const data = await response.json();
      setResponse(data.response);
    } catch (err) {
      console.error('Error querying API:', err);
      setError('Failed to get a response. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleMachineChange = (e) => {
    const machineId = e.target.value;
    const machine = machines.find(m => m.id === machineId);
    setSelectedMachine(machine || null);
    setResponse(''); // Clear previous responses when changing machines
  };
  
  // Format the response with proper styling for citations
  const formatResponse = (text) => {
    if (!text) return '';
    
    // Replace citations with styled spans
    return text.split(/(\[Document: [^\]]+\])/).map((part, index) => {
      if (part.match(/\[Document: [^\]]+\]/)) {
        return <span key={index} className="text-blue-600 font-medium">{part}</span>;
      }
      return part;
    });
  };

  return (
    <div className="flex flex-col h-screen max-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Book size={24} />
            <h1 className="text-xl font-bold">Service Manual Assistant</h1>
          </div>
          
          {/* Machine Selection */}
          <div className="flex items-center space-x-2">
            <select
              className="px-3 py-2 rounded bg-white text-gray-800 border-none w-64"
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
              className="p-2 bg-blue-700 rounded hover:bg-blue-800"
              disabled={machinesLoading}
            >
              {machinesLoading ? (
                <Loader2 size={20} className="animate-spin" />
              ) : (
                <RefreshCcw size={20} />
              )}
            </button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="flex-1 container mx-auto p-4 flex flex-col overflow-hidden">
        {/* Machine Info */}
        {selectedMachine && (
          <div className="bg-blue-50 p-4 rounded-lg mb-4 border border-blue-100">
            <h2 className="text-lg font-semibold text-blue-800">
              {selectedMachine.name} ({selectedMachine.model})
            </h2>
            {selectedMachine.description && (
              <p className="text-sm text-blue-600 mt-1">{selectedMachine.description}</p>
            )}
          </div>
        )}
        
        {/* Error Message */}
        {error && (
          <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-4 border border-red-100">
            {error}
          </div>
        )}
        
        {/* Response Area */}
        <div 
          ref={responseRef}
          className="flex-1 bg-white rounded-lg shadow-md p-6 mb-4 overflow-y-auto"
        >
          {response ? (
            <div className="prose max-w-none">
              <div className="whitespace-pre-line">{formatResponse(response)}</div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400">
              {selectedMachine ? (
                <p>Ask a question about the {selectedMachine.name} service manual</p>
              ) : (
                <p>Please select a machine first</p>
              )}
            </div>
          )}
          
          {loading && (
            <div className="flex justify-center mt-4">
              <Loader2 size={40} className="animate-spin text-blue-600" />
            </div>
          )}
        </div>
        
        {/* Query Input */}
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about the service manual..."
            className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading || !selectedMachine}
          />
          <button
            type="submit"
            className="bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-400"
            disabled={loading || !query.trim() || !selectedMachine}
          >
            {loading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </form>
      </main>
    </div>
  );
};

export default ServiceManualRAG;
