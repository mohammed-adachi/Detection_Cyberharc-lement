"use client";

import { useState, useEffect } from "react";
import { MagnifyingGlassIcon, FunnelIcon } from "@heroicons/react/24/outline";
const Messages = () => {
  const [filter, setFilter] = useState("all");
  const [messages, setMessages] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await fetch("http://localhost:5000/messages");
        console.log(response);
        if (response.ok) {
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            setMessages(data);
          } else {
            throw new Error("La réponse du serveur n'est pas au format JSON");
          }
        } else {
          throw new Error(`Erreur HTTP: ${response.status}`);
        }
      } catch (error) {
        console.error("Error fetching messages:", error);
        setError(
          `Erreur lors de la récupération des messages: ${error.message}`
        );
      }
    };

    fetchMessages();
  }, []);

  const filteredMessages = messages.filter((message) => {
    if (
      filter === "harassing" &&
      message.cyberbullying_type !== "other_cyberbullying"
    ) {
      return false;
    }
    if (
      filter === "non-harassing" &&
      message.cyberbullying_type === "not_cyberbullying"
    ) {
      return false;
    }
    return message.message.toLowerCase().includes(searchTerm.toLowerCase());
  });

  if (error) {
    return (
      <div className="text-red-500 p-4 bg-red-100 rounded-md">
        <h2 className="text-xl font-bold mb-2">Erreur</h2>
        <p>{error}</p>
        <p className="mt-2">
          Veuillez vérifier que le serveur fonctionne correctement et que l'API
          renvoie des données JSON valides.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">Messages</h2>

      <div className="flex justify-between items-center mb-6">
        <div className="flex space-x-2">
          <button
            onClick={() => setFilter("all")}
            className={`px-4 py-2 rounded ${
              filter === "all" ? "bg-blue-500 text-white" : "bg-gray-200"
            }`}
          >
            Tous
          </button>
          <button
            onClick={() => setFilter("harassing")}
            className={`px-4 py-2 rounded ${
              filter === "harassing" ? "bg-blue-500 text-white" : "bg-gray-200"
            }`}
          >
            other_cyberbullying
          </button>
          <button
            onClick={() => setFilter("non-harassing")}
            className={`px-4 py-2 rounded ${
              filter === "non-harassing"
                ? "bg-blue-500 text-white"
                : "bg-gray-200"
            }`}
          >
            not_cyberbullying
          </button>
        </div>
        <div className="flex items-center space-x-2">
          <div className="relative">
            <input
              type="text"
              placeholder="Rechercher..."
              className="pl-10 pr-4 py-2 border rounded-full"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
          </div>
          <button className="p-2 bg-gray-200 rounded-full">
            <FunnelIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Message
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Type de cyberharcèlement
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Statut
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredMessages.map((message) => (
              <tr key={message._id} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  <div className="flex items-center">
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {message.message}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {message.cyberbullying_type}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      message.cyberbullying_type === "Harassment"
                        ? "bg-red-100 text-red-800"
                        : "bg-green-100 text-green-800"
                    }`}
                  >
                    {message.cyberbullying_type === "Harassment"
                      ? "Harcelant"
                      : "Non Harcelant"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Messages;
