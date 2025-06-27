import React from 'react';
import { FiCircle, FiCheckCircle, FiLoader } from 'react-icons/fi';

const ProgressTracker = ({ progressData, auditPlan }) => {
  // Use audit plan sections if available, otherwise fallback to progress data
  const sections = auditPlan?.sections || progressData?.section_progress || [];
  const completionPercentage = progressData?.completion_percentage || 0;

  if (!sections.length) {
    return null;
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <FiCheckCircle className="w-5 h-5 text-green-600" />;
      case 'in_progress':
        return <FiLoader className="w-5 h-5 text-blue-600 animate-spin" />;
      default:
        return <FiCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'in_progress':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-600';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Audit Progress</h3>
        <span className="text-sm font-medium text-gray-600">{completionPercentage}%</span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
        <div 
          className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${completionPercentage}%` }}
        />
      </div>

      <div className="space-y-3">
        {sections.map((section, index) => {
          const sectionName = section.name || section.cycle?.replace('_', ' ') || `Section ${index + 1}`;
          const status = section.status || 'pending';
          
          return (
            <div
              key={section.id || index}
              className={`flex items-center space-x-3 p-3 rounded-lg border ${getStatusColor(status)}`}
            >
              {getStatusIcon(status)}
              <span className="flex-1 text-sm font-medium">{sectionName}</span>
              <span className="text-xs opacity-75">{index + 1}</span>
            </div>
          );
        })}      </div>
    </div>
  );
};

export default ProgressTracker;
