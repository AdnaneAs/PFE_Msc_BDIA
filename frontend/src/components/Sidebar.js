import React from 'react';
import { FiHome, FiUpload, FiFileText, FiSearch, FiSettings } from 'react-icons/fi';

const Sidebar = ({ onSectionChange, activeSection }) => {
  const navItems = [
    { key: 'home', label: 'Home', icon: <FiHome /> },
    { key: 'upload', label: 'Upload', icon: <FiUpload /> },
    { key: 'documents', label: 'Documents', icon: <FiFileText /> },
    { key: 'query', label: 'Query', icon: <FiSearch /> },
    { key: 'settings', label: 'Settings', icon: <FiSettings /> },
  ];

  return (
    <aside className="fixed top-0 left-0 h-full w-20 bg-gradient-to-b from-purple-700 via-purple-500 to-white shadow-xl flex flex-col items-center py-8 z-30">
      <div className="mb-10">
        <span className="text-3xl font-extrabold text-white drop-shadow">RAG</span>
      </div>
      <nav className="flex flex-col gap-8 w-full items-center">
        {navItems.map(item => (
          <button
            key={item.key}
            onClick={() => onSectionChange(item.key)}
            className={`flex flex-col items-center text-xl p-2 rounded-xl transition-all duration-300 w-16 h-16 mb-2
              ${activeSection === item.key ? 'bg-white/80 text-purple-700 shadow-lg scale-110' : 'text-white hover:bg-white/30 hover:text-purple-200'}`}
            title={item.label}
          >
            {item.icon}
            <span className="text-xs mt-1 font-semibold">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
