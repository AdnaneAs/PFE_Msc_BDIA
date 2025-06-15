import React from 'react';

const FullScreenLayout = ({ children }) => (
  <div className="w-full h-full min-h-screen min-w-0 flex flex-col bg-gradient-to-br from-purple-100 via-white to-purple-200">
    {children}
  </div>
);

export default FullScreenLayout;
