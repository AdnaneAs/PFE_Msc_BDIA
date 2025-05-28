import React, { useState, useEffect } from 'react';
import { FiWifi, FiWifiOff, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';

const ConnectionStatus = () => {
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [lastCheck, setLastCheck] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  // Check backend connection
  const checkConnection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('http://localhost:8000/api/hello', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('error');
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        setConnectionStatus('timeout');
      } else {
        setConnectionStatus('disconnected');
      }
    }
    
    setLastCheck(new Date());
  };

  // Check connection on mount and periodically
  useEffect(() => {
    checkConnection();
    
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusInfo = () => {
    switch (connectionStatus) {
      case 'connected':
        return {
          icon: <FiCheckCircle className="text-green-500" />,
          text: 'Connected',
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200'
        };
      case 'disconnected':
        return {
          icon: <FiWifiOff className="text-red-500" />,
          text: 'Disconnected',
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200'
        };
      case 'timeout':
        return {
          icon: <FiAlertTriangle className="text-yellow-500" />,
          text: 'Slow Connection',
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200'
        };
      case 'error':
        return {
          icon: <FiAlertTriangle className="text-orange-500" />,
          text: 'Server Error',
          color: 'text-orange-600',
          bgColor: 'bg-orange-50',
          borderColor: 'border-orange-200'
        };
      case 'checking':
      default:
        return {
          icon: <FiWifi className="text-blue-500 animate-pulse" />,
          text: 'Checking...',
          color: 'text-blue-600',
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200'
        };
    }
  };

  const statusInfo = getStatusInfo();

  // Only show if there's an issue or user wants to see details
  if (connectionStatus === 'connected' && !showDetails) {
    return (
      <button
        onClick={() => setShowDetails(true)}
        className="text-xs text-gray-500 hover:text-gray-700 flex items-center"
        title="Show connection status"
      >
        {statusInfo.icon}
        <span className="ml-1">Backend</span>
      </button>
    );
  }

  return (
    <div className={`fixed top-4 right-4 z-50 max-w-sm ${showDetails ? 'block' : ''}`}>
      <div className={`${statusInfo.bgColor} ${statusInfo.borderColor} border rounded-lg p-3 shadow-lg`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {statusInfo.icon}
            <span className={`ml-2 font-medium text-sm ${statusInfo.color}`}>
              Backend Server: {statusInfo.text}
            </span>
          </div>
          {showDetails && connectionStatus === 'connected' && (
            <button
              onClick={() => setShowDetails(false)}
              className="ml-2 text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          )}
        </div>
        
        {/* Show details for problematic connections */}
        {connectionStatus !== 'connected' && connectionStatus !== 'checking' && (
          <div className="mt-2 text-xs text-gray-600">
            <div className="space-y-1">
              {connectionStatus === 'disconnected' && (
                <>
                  <p>• Check if the backend server is running</p>
                  <p>• Ensure server is accessible at http://localhost:8000</p>
                </>
              )}
              {connectionStatus === 'timeout' && (
                <>
                  <p>• Server is responding slowly</p>
                  <p>• Requests may take longer than usual</p>
                </>
              )}
              {connectionStatus === 'error' && (
                <>
                  <p>• Server returned an error</p>
                  <p>• Check server logs for details</p>
                </>
              )}
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-200">
                <button
                  onClick={checkConnection}
                  className="text-blue-600 hover:text-blue-800 underline"
                >
                  Retry Connection
                </button>
                {lastCheck && (
                  <span className="text-gray-400">
                    Last check: {lastCheck.toLocaleTimeString()}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Show details for connected state when requested */}
        {connectionStatus === 'connected' && showDetails && (
          <div className="mt-2 text-xs text-gray-600">
            <p>✓ Backend server is responding normally</p>
            <p>✓ API endpoints are accessible</p>
            {lastCheck && (
              <p className="mt-1">Last verified: {lastCheck.toLocaleTimeString()}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ConnectionStatus;
