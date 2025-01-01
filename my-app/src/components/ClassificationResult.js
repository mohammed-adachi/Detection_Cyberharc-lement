import React from 'react';

const ClassificationResult = ({
  message,
  cyberbullying_type,
  probabilities_by_class,
  max_probability
}) => {
  return (
    <div className="mt-4 p-4 bg-gray-100 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">Résultat de la classification :</h3>
      <p><strong>Message :</strong> {message}</p>
      <p><strong>Type de cyberharcèlement :</strong> {cyberbullying_type}</p>
      <p><strong>Probabilités par classe :</strong></p>
      {probabilities_by_class ? (
        <ul className="list-disc list-inside ml-4">
          {Object.entries(probabilities_by_class).map(([key, value]) => (
            <li key={key}>{key}: {value.toFixed(4)}</li>
          ))}
        </ul>
      ) : (
        <p>Aucune probabilité disponible</p>
      )}
      <p><strong>Probabilité maximale :</strong> {max_probability.toFixed(4)}</p>
    </div>
  );
};

export default ClassificationResult;
