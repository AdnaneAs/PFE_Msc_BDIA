import React, { useState, useEffect } from 'react';
import { 
  FiX, FiCheckCircle, FiXCircle, FiAlertCircle, FiInfo, FiBell, 
  FiRefreshCw, FiActivity, FiClock, FiTrendingUp, FiZap, FiCpu 
} from 'react-icons/fi';

const NotificationSystem = ({ notifications, removeNotification }) => {
  return (
    <div className="notification-container">
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onRemove={removeNotification}
        />
      ))}
    </div>
  );
};

const NotificationItem = ({ notification, onRemove }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  const [progressWidth, setProgressWidth] = useState(100);

  useEffect(() => {
    // Trigger entrance animation with slight delay for smooth effect
    const timer = setTimeout(() => setIsVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Auto-dismiss with progress animation
    if (notification.duration && notification.duration > 0) {
      const startTime = Date.now();
      
      const updateProgress = () => {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, notification.duration - elapsed);
        const progress = (remaining / notification.duration) * 100;
        
        setProgressWidth(progress);
        
        if (remaining > 0) {
          requestAnimationFrame(updateProgress);
        } else {
          handleRemove();
        }
      };
      
      requestAnimationFrame(updateProgress);
    }
  }, [notification.duration]);

  const handleRemove = () => {
    setIsLeaving(true);
    setTimeout(() => {
      onRemove(notification.id);
    }, 400); // Wait for exit animation
  };
  const getIcon = () => {
    const iconProps = { size: 20, className: "notification-icon" };
    
    switch (notification.type) {
      case 'success':
        return <FiCheckCircle {...iconProps} className="notification-icon text-green-600" />;
      case 'error':
        return <FiXCircle {...iconProps} className="notification-icon text-red-600" />;
      case 'warning':
        return <FiAlertCircle {...iconProps} className="notification-icon text-yellow-600" />;
      case 'processing':
        return <FiRefreshCw {...iconProps} className="notification-icon text-purple-600" />;
      case 'workflow':
        return <FiActivity {...iconProps} className="notification-icon text-blue-600" />;
      case 'status':
        return <FiClock {...iconProps} className="notification-icon text-orange-600" />;
      case 'progress-update':
        return <FiTrendingUp {...iconProps} className="notification-icon text-green-600" />;
      case 'backend-state':
        return <FiCpu {...iconProps} className="notification-icon text-indigo-600" />;
      case 'performance':
        return <FiZap {...iconProps} className="notification-icon text-yellow-500" />;
      case 'info':
      default:
        return <FiInfo {...iconProps} className="notification-icon text-blue-600" />;
    }
  };
  const getNotificationClasses = () => {
    const baseClasses = "notification-item notification-glass rounded-lg p-4 transition-all duration-400 ease-out transform";
    
    const typeClasses = {
      success: "notification-success",
      error: "notification-error", 
      warning: "notification-warning",
      info: "notification-info",
      processing: "notification-processing",
      workflow: "notification-workflow",
      status: "notification-status",
      'progress-update': "notification-progress-update",
      'backend-state': "notification-processing",
      performance: "notification-warning"
    };
    
    const animationClasses = isVisible && !isLeaving 
      ? "translate-x-0 opacity-100 scale-100" 
      : "translate-x-full opacity-0 scale-95";
    
    return `${baseClasses} ${typeClasses[notification.type] || typeClasses.info} ${animationClasses}`;
  };
  const getProgressColor = () => {
    const colors = {
      success: "#10B981",
      error: "#EF4444", 
      warning: "#F59E0B",
      info: "#3B82F6",
      processing: "#8B5CF6",
      workflow: "#6366F1",
      status: "#F59E0B",
      'progress-update': "#10B981",
      'backend-state': "#8B5CF6",
      performance: "#F59E0B"
    };
    return colors[notification.type] || colors.info;
  };
  return (
    <div className={getNotificationClasses()}>
      {/* Enhanced progress bar with smooth animation */}
      {notification.duration && notification.duration > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-black bg-opacity-10 rounded-b-lg overflow-hidden">
          <div 
            className="notification-progress h-full transition-all duration-100 ease-linear"
            style={{
              width: `${progressWidth}%`,
              backgroundColor: getProgressColor(),
            }}
          />
        </div>
      )}

      {/* Main content with modern layout */}
      <div className="flex items-start space-x-3">
        {/* Icon with enhanced styling */}
        <div className="flex-shrink-0 pt-0.5">
          {getIcon()}
        </div>
        
        {/* Content area */}
        <div className="flex-1 min-w-0">
          {notification.title && (
            <h4 className="notification-title">
              {notification.title}
            </h4>
          )}
          <p className="notification-message">
            {notification.message}
          </p>
          
          {/* Optional timestamp */}
          {notification.showTimestamp && (
            <p className="text-xs text-gray-500 mt-1 opacity-75">
              {new Date().toLocaleTimeString()}
            </p>
          )}
        </div>
        
        {/* Enhanced close button */}
        <button
          onClick={handleRemove}
          className="notification-close flex-shrink-0 text-gray-400 hover:text-gray-700"
          aria-label="Close notification"
        >
          <FiX size={16} />
        </button>
      </div>

      {/* Subtle animated border effect */}
      <div className="absolute inset-0 rounded-lg opacity-30 pointer-events-none overflow-hidden">
        <div 
          className="absolute inset-0 animate-pulse"
          style={{
            background: `linear-gradient(135deg, transparent 0%, ${getProgressColor()}15 50%, transparent 100%)`,
            animation: 'shimmer 3s ease-in-out infinite'
          }}
        />
      </div>
    </div>
  );
};

// Enhanced hook for managing notifications with modern features
export const useNotifications = () => {
  const [notifications, setNotifications] = useState([]);

  const addNotification = (notification) => {
    const id = Date.now() + Math.random();
    const newNotification = {
      id,
      duration: 4000, // Default 4 seconds for better UX
      showTimestamp: false,
      ...notification,
    };
    
    setNotifications(prev => [...prev, newNotification]);
    
    // Add subtle sound effect (optional)
    if (notification.playSound !== false) {
      try {
        // Create a subtle notification sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(notification.type === 'error' ? 300 : 800, audioContext.currentTime);
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.2);
      } catch (e) {
        // Ignore audio errors
      }
    }
    
    return id;
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const clearAllNotifications = () => {
    setNotifications([]);
  };

  // Enhanced helper methods with better defaults
  const showSuccess = (message, options = {}) => {
    return addNotification({ 
      type: 'success', 
      message, 
      title: options.title || '✓ Success', 
      duration: options.duration || 3500,
      showTimestamp: options.showTimestamp || false,
      ...options 
    });
  };

  const showError = (message, options = {}) => {
    return addNotification({ 
      type: 'error', 
      message, 
      title: options.title || '✗ Error', 
      duration: options.duration || 6000, // Longer for errors
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showWarning = (message, options = {}) => {
    return addNotification({ 
      type: 'warning', 
      message, 
      title: options.title || '⚠ Warning', 
      duration: options.duration || 5000,
      showTimestamp: options.showTimestamp || false,
      ...options 
    });
  };

  const showInfo = (message, options = {}) => {
    return addNotification({ 
      type: 'info', 
      message, 
      title: options.title || 'ℹ Information', 
      duration: options.duration || 4000,
      showTimestamp: options.showTimestamp || false,
      ...options 
    });
  };
  const showProcessing = (message, options = {}) => {
    return addNotification({ 
      type: 'processing', 
      message, 
      title: options.title || '⟳ Processing', 
      duration: options.duration || 0, // No auto-dismiss for processing
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showWorkflow = (message, options = {}) => {
    return addNotification({ 
      type: 'workflow', 
      message, 
      title: options.title || '🔄 Workflow Update', 
      duration: options.duration || 4000,
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showStatus = (message, options = {}) => {
    return addNotification({ 
      type: 'status', 
      message, 
      title: options.title || '📊 Status Update', 
      duration: options.duration || 3000,
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showProgressUpdate = (message, options = {}) => {
    return addNotification({ 
      type: 'progress-update', 
      message, 
      title: options.title || '📈 Progress Update', 
      duration: options.duration || 3500,
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showBackendState = (message, options = {}) => {
    return addNotification({ 
      type: 'backend-state', 
      message, 
      title: options.title || '🖥️ Backend Status', 
      duration: options.duration || 0, // No auto-dismiss for backend states
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  const showPerformance = (message, options = {}) => {
    return addNotification({ 
      type: 'performance', 
      message, 
      title: options.title || '⚡ Performance Alert', 
      duration: options.duration || 5000,
      showTimestamp: options.showTimestamp || true,
      ...options 
    });
  };

  return {
    notifications,
    addNotification,
    removeNotification,
    clearAllNotifications,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showProcessing,
    showWorkflow,
    showStatus,
    showProgressUpdate,
    showBackendState,
    showPerformance,
  };
};

export default NotificationSystem;
