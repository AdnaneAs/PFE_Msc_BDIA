import React, { useState, useEffect } from 'react';
import { FiWifi, FiWifiOff, FiAlertTriangle, FiCheckCircle, FiRefreshCw, FiCpu, FiZap } from 'react-icons/fi';

// Import NotificationSystem safely - if it fails, we'll use fallback notifications
let useNotifications = null;
try {
  const NotificationModule = require('./NotificationSystem');
  useNotifications = NotificationModule.useNotifications;
} catch (error) {
  console.warn('NotificationSystem not available, using fallback notifications');
}

const ConnectionStatus = () => {
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [lastCheck, setLastCheck] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  // Use the notification system if available
  const notifications = useNotifications ? useNotifications() : {
    showBackendState: () => null,
    showPerformance: () => null,
    showError: () => null,
    showSuccess: () => null,
    removeNotification: () => null
  };

  const {
    showBackendState,
    showPerformance,
    showError,
    showSuccess,
    removeNotification
  } = notifications;

  // Track notification IDs to avoid duplicates
  const [lastNotificationId, setLastNotificationId] = useState(null);
  // Check backend connection
  const checkConnection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('http://localhost:8000/api/hello', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      const newStatus = response.ok ? 'connected' : 'error';
      const previousStatus = connectionStatus;
      
      setConnectionStatus(newStatus);
      
      // Show notifications for status changes
      if (previousStatus !== newStatus && previousStatus !== 'checking') {
        // Clear previous notification
        if (lastNotificationId) {
          removeNotification(lastNotificationId);
        }
        
        if (newStatus === 'connected' && previousStatus !== 'connected') {
          const notificationId = showSuccess('Backend server is now responding normally', {
            title: '✅ Connection Restored',
            duration: 4000
          });
          setLastNotificationId(notificationId);
        } else if (newStatus === 'error') {
          const notificationId = showError('Backend server returned an error - check server logs', {
            title: '🚫 Server Error',
            duration: 8000,
            showTimestamp: true
          });
          setLastNotificationId(notificationId);
        }
      }
      
    } catch (error) {
      const previousStatus = connectionStatus;
      let newStatus;
      let notificationId;
      
      if (error.name === 'AbortError') {
        newStatus = 'timeout';
        if (previousStatus !== 'timeout' && previousStatus !== 'checking') {
          // Clear previous notification
          if (lastNotificationId) {
            removeNotification(lastNotificationId);
          }
          
          notificationId = showPerformance('Backend server is responding slowly - requests may take longer than usual', {
            title: '⏱️ Slow Connection',
            showTimestamp: true
          });
          setLastNotificationId(notificationId);
        }
      } else {
        newStatus = 'disconnected';
        if (previousStatus !== 'disconnected' && previousStatus !== 'checking') {
          // Clear previous notification
          if (lastNotificationId) {
            removeNotification(lastNotificationId);
          }
          
          notificationId = showBackendState('Backend server is not available - check if server is running on http://localhost:8000', {
            title: '🖥️ Connection Lost',
            showTimestamp: true
          });
          setLastNotificationId(notificationId);
        }
      }
      
      setConnectionStatus(newStatus);
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
          icon: <FiCheckCircle className="text-green-500 notification-icon" />,
          text: 'Connected',
          color: 'text-green-600',
          bgClass: 'notification-success',
          iconColor: 'text-green-600'
        };
      case 'disconnected':
        return {
          icon: <FiCpu className="text-red-500 notification-icon" />,
          text: 'Disconnected',
          color: 'text-red-600',
          bgClass: 'notification-error',
          iconColor: 'text-red-600'
        };
      case 'timeout':
        return {
          icon: <FiZap className="text-yellow-500 notification-icon" />,
          text: 'Slow Connection',
          color: 'text-yellow-600',
          bgClass: 'notification-performance',
          iconColor: 'text-yellow-600'
        };
      case 'error':
        return {
          icon: <FiAlertTriangle className="text-orange-500 notification-icon" />,
          text: 'Server Error',
          color: 'text-orange-600',
          bgClass: 'notification-error',
          iconColor: 'text-orange-600'
        };
      case 'checking':
      default:
        return {
          icon: <FiRefreshCw className="text-purple-500 animate-spin notification-icon" />,
          text: 'Checking...',
          color: 'text-purple-600',
          bgClass: 'notification-processing',
          iconColor: 'text-purple-600'
        };
    }
  };

  const statusInfo = getStatusInfo();
  // Only show if there's an issue or user wants to see details
  if (connectionStatus === 'connected' && !showDetails) {
    return (      <button
        onClick={() => setShowDetails(true)}
        className="fixed top-4 left-20 z-50 text-xs text-gray-500 hover:text-gray-700 
                   flex items-center px-3 py-2 rounded-lg bg-white shadow-md 
                   hover:shadow-lg transition-all duration-200 border border-gray-200"
        title="Show connection status"
      >
        {statusInfo.icon}
        <span className="ml-2 font-medium">Backend</span>
      </button>
    );
  }

  return (
    <div className="fixed top-6 left-30 z-50 max-w-sm">
      <div className={`notification-glass rounded-lg p-4 shadow-lg transition-all duration-400 
                      transform hover:scale-105 ${statusInfo.bgClass} backdrop-filter backdrop-blur-lg`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
              {statusInfo.icon}
            </div>
            <div>
              <h4 className="notification-title">
                Backend Server: {statusInfo.text}
              </h4>
              {lastCheck && (
                <p className="notification-message text-xs opacity-75">
                  Last check: {lastCheck.toLocaleTimeString()}
                </p>
              )}
            </div>
          </div>
          {showDetails && connectionStatus === 'connected' && (
            <button
              onClick={() => setShowDetails(false)}
              className="notification-close ml-3 text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          )}
        </div>
        
        {/* Enhanced progress indicator for processing states */}
        {connectionStatus === 'checking' && (
          <div className="notification-progress mt-3"></div>
        )}
        
        {/* Show details for problematic connections */}
        {connectionStatus !== 'connected' && connectionStatus !== 'checking' && (          <div className="mt-4 space-y-2">
            <div className="space-y-1 text-sm text-white">
              {connectionStatus === 'disconnected' && (
                <>
                  <p className="flex items-center text-white">
                    <FiCpu className="mr-2 text-red-500" />
                    Check if the backend server is running
                  </p>
                  <p className="flex items-center text-white">
                    <FiWifi className="mr-2 text-red-500" />
                    Ensure server is accessible at http://localhost:8000
                  </p>
                </>
              )}
              {connectionStatus === 'timeout' && (
                <>
                  <p className="flex items-center text-white">
                    <FiZap className="mr-2 text-yellow-500" />
                    Server is responding slowly
                  </p>
                  <p className="flex items-center text-white">
                    <FiAlertTriangle className="mr-2 text-yellow-500" />
                    Requests may take longer than usual
                  </p>
                </>
              )}
              {connectionStatus === 'error' && (
                <>
                  <p className="flex items-center text-white">
                    <FiAlertTriangle className="mr-2 text-orange-500" />
                    Server returned an error
                  </p>
                  <p className="flex items-center text-white">
                    <FiCpu className="mr-2 text-orange-500" />
                    Check server logs for details
                  </p>
                </>
              )}
            </div>
            
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-white border-opacity-20">
              <button
                onClick={checkConnection}
                className="flex items-center space-x-2 px-4 py-2 bg-white bg-opacity-20 
                         rounded-lg hover:bg-opacity-30 transition-all duration-200 
                         text-sm font-medium backdrop-filter backdrop-blur-sm"
              >
                <FiRefreshCw className="w-4 h-4" />
                <span>Retry Connection</span>
              </button>
            </div>
          </div>
        )}
        
        {/* Show details for connected state when requested */}
        {connectionStatus === 'connected' && showDetails && (          <div className="mt-4 space-y-2 text-sm text-white">
            <p className="flex items-center text-white">
              <FiCheckCircle className="mr-2 text-green-500" />
              Backend server is responding normally
            </p>
            <p className="flex items-center text-white">
              <FiWifi className="mr-2 text-green-500" />
              API endpoints are accessible
            </p>
            {lastCheck && (
              <p className="flex items-center opacity-75 text-white">
                <FiCpu className="mr-2 text-green-500" />
                Last verified: {lastCheck.toLocaleTimeString()}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ConnectionStatus;
