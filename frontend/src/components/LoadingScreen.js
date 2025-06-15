import React from 'react';
import { motion } from 'framer-motion';

const glassBg = {
  background: 'rgba(255,255,255,0.15)',
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  backdropFilter: 'blur(12px)',
  WebkitBackdropFilter: 'blur(12px)',
  borderRadius: '24px',
  border: '1px solid rgba(255,255,255,0.18)'
};

const LoadingScreen = () => {
  return (
    <div className="fixed inset-0 flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-700 via-purple-400 to-white overflow-hidden">
      {/* Animated flowing glass shapes */}
      <motion.div
        className="absolute w-[600px] h-[600px] left-[-200px] top-[-200px]"
        style={{ ...glassBg, background: 'rgba(168,85,247,0.25)' }}
        animate={{ x: [0, 100, 0], y: [0, 100, 0], rotate: [0, 30, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute w-[400px] h-[400px] right-[-100px] bottom-[-100px]"
        style={{ ...glassBg, background: 'rgba(255,255,255,0.18)' }}
        animate={{ x: [0, -80, 0], y: [0, -80, 0], rotate: [0, -20, 0] }}
        transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
      />
      <div className="relative z-10 flex flex-col items-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1 }}
          className="mb-8"
        >
          <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
            <circle cx="40" cy="40" r="36" stroke="#a855f7" strokeWidth="8" fill="white" fillOpacity="0.7" />
            <circle cx="40" cy="40" r="24" stroke="#a855f7" strokeWidth="4" fill="none" />
          </svg>
        </motion.div>
        <motion.h1
          className="text-4xl font-extrabold text-white drop-shadow-lg mb-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          Welcome to Enterprise Audit Report Generator
        </motion.h1>
        <motion.p
          className="text-lg text-purple-100 mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          Loading, please wait...
        </motion.p>
        <motion.div
          className="w-24 h-2 rounded-full bg-purple-200 overflow-hidden"
          initial={{ width: 0 }}
          animate={{ width: '6rem' }}
          transition={{ duration: 1.2, repeat: Infinity, repeatType: 'reverse' }}
        >
          <motion.div
            className="h-2 bg-purple-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: '100%' }}
            transition={{ duration: 1.2, repeat: Infinity, repeatType: 'reverse' }}
          />
        </motion.div>
      </div>
    </div>
  );
};

export default LoadingScreen;
