import React from 'react';

function DiagnosticApp() {
  return (
    <div style={{ 
      backgroundColor: 'red', 
      color: 'white', 
      padding: '50px', 
      margin: '50px',
      fontSize: '24px',
      border: '5px solid blue'
    }}>
      <h1>DIAGNOSTIC TEST - If you see this, React is working!</h1>
      <p>Date: {new Date().toISOString()}</p>
      <p>This should be very visible with red background and blue border.</p>
    </div>
  );
}

export default DiagnosticApp;
