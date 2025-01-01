import React from 'react';

const DetectionResult = ({ result }) => {
  if (result === null) {
    return null;
  }

  const resultText = result ? 'Le message contient "not_building"' : 'Le message ne contient pas "not_building"';
  const resultClass = result ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800';

  return React.createElement('div', { className: `p-4 rounded ${resultClass}` },
    resultText
  );
};

export default DetectionResult;