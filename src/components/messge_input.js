// import React, { useState, useCallback } from "react";

// const MessageInput = ({ onSubmit }) => {
//   const [message, setMessage] = useState("");
//   const [isLoading, setIsLoading] = useState(false);

//   const handleSubmit = useCallback(
//     async (e) => {
//       e.preventDefault();
//       setIsLoading(true);
//       try {
//         console.log("Message:", message);
//         const response = await fetch("http://localhost:5000/add_message", {
//           method: "POST",
//           headers: {
//             "Content-Type": "application/json",
//           },
//           body: JSON.stringify({
//             message: message,
//             cyberbullying_type: "not_bullying",
//           }),
//         });

//         if (!response.ok) {
//           throw new Error("Network response was not ok");
//         }

//         const result = await response.json();
//         console.log("Message added:", result);
//         onSubmit(message);
//         setMessage("");
//       } catch (error) {
//         console.error("Error:", error);
//       } finally {
//         setIsLoading(false);
//       }
//     },
//     [message, onSubmit]
//   );

//   return React.createElement(
//     "form",
//     { onSubmit: handleSubmit, className: "mb-4" },
//     React.createElement("input", {
//       type: "text",
//       value: message,
//       onChange: (e) => setMessage(e.target.value),
//       placeholder: "Tapez votre message ici",
//       className: "w-full p-2 border border-gray-300 rounded",
//     }),
//     React.createElement(
//       "button",
//       {
//         type: "submit",
//         className:
//           "mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300",
//         disabled: isLoading,
//       },
//       isLoading ? "Envoi..." : "Envoyer"
//     )
//   );
// };

// export default MessageInput;
import React, { useState, useCallback } from "react";
import Button from "@mui/material/Button";
import  CircularProgress  from "@mui/material/CircularProgress";
import  Card  from "@mui/material/Card";
import  CardContent  from "@mui/material/CardContent";
import  CardHeader  from "@mui/material/CardHeader";
import  TextField  from "@mui/material/TextField";
import  Typography  from "@mui/material/Typography";
import  {CheckCircle, ErrorOutline } from "@mui/icons-material";

const MessageInput = ({ onSubmit }) => {
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setIsLoading(true);
      try {
        // POST request to add the message
        const postResponse = await fetch("http://localhost:5000/add_message", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
            cyberbullying_type: "not_bullying",
          }),
        });

        if (!postResponse.ok) {
          throw new Error("Network response was not ok for POST request");
        }

        const postResult = await postResponse.json();
        console.log("Message added:", postResult);

        // GET request to analyze the message
        const getResponse = await fetch(
          `http://localhost:5000/message/${encodeURIComponent(message)}`
        );

        if (!getResponse.ok) {
          throw new Error("Network response was not ok for GET request");
        }

        const analysisResult = await getResponse.json();
        console.log("Message analysis:", analysisResult);
        setAnalysis(analysisResult);

        onSubmit(message);
        setMessage("");
      } catch (error) {
        console.error("Error:", error);
      } finally {
        setIsLoading(false);
      }
    },
    [message, onSubmit]
  );

  return (
    <div className="max-w-2xl mx-auto p-4">
      <Card className="mb-6">
        <CardHeader>
          <Typography variant="h6">Analyseur de Messages</Typography>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <TextField
              label="Tapez votre message ici"
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              fullWidth
              variant="outlined"
              disabled={isLoading}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              disabled={isLoading}
            >
              {isLoading ? <CircularProgress size={24} /> : "Envoyer"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {analysis && (
        <Card>
          <CardHeader>
            <Typography variant="h6" className="flex items-center">
              <span className="mr-2">Analyse du message</span>
              {analysis.cyberbullying_type === "not_cyberbullying" ? (
                <CheckCircle style={{ color: "green" }} />
              ) : (
                <ErrorOutline style={{ color: "red" }} />
              )}
            </Typography>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>Message :</strong> {analysis.message}
              </p>
              <p>
                <strong>Type de cyberharcèlement :</strong>{" "}
                {analysis.cyberbullying_type}
              </p>
              <p>
                <strong>Probabilité maximale :</strong>{" "}
                {(analysis.max_probability * 100).toFixed(2)}%
              </p>
              <div>
                <strong>Probabilités par classe :</strong>
                <ul className="list-disc list-inside pl-4 mt-1">
                  <li>
                    Non-cyberharcèlement :{" "}
                    {(
                      analysis.probabilities_by_class.not_cyberbullying * 100
                    ).toFixed(2)}
                    %
                  </li>
                  <li>
                    Autre cyberharcèlement :{" "}
                    {(
                      analysis.probabilities_by_class.other_cyberbullying * 100
                    ).toFixed(2)}
                    %
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default MessageInput;
