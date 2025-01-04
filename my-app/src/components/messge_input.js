// // import React, { useState, useCallback } from "react";

// // const MessageInput = ({ onSubmit }) => {
// //   const [message, setMessage] = useState("");
// //   const [isLoading, setIsLoading] = useState(false);

// //   const handleSubmit = useCallback(
// //     async (e) => {
// //       e.preventDefault();
// //       setIsLoading(true);
// //       try {
// //         console.log("Message:", message);
// //         const response = await fetch("http://localhost:5000/add_message", {
// //           method: "POST",
// //           headers: {
// //             "Content-Type": "application/json",
// //           },
// //           body: JSON.stringify({
// //             message: message,
// //             cyberbullying_type: "not_bullying",
// //           }),
// //         });

// //         if (!response.ok) {
// //           throw new Error("Network response was not ok");
// //         }

// //         const result = await response.json();
// //         console.log("Message added:", result);
// //         onSubmit(message);
// //         setMessage("");
// //       } catch (error) {
// //         console.error("Error:", error);
// //       } finally {
// //         setIsLoading(false);
// //       }
// //     },
// //     [message, onSubmit]
// //   );

// //   return React.createElement(
// //     "form",
// //     { onSubmit: handleSubmit, className: "mb-4" },
// //     React.createElement("input", {
// //       type: "text",
// //       value: message,
// //       onChange: (e) => setMessage(e.target.value),
// //       placeholder: "Tapez votre message ici",
// //       className: "w-full p-2 border border-gray-300 rounded",
// //     }),
// //     React.createElement(
// //       "button",
// //       {
// //         type: "submit",
// //         className:
// //           "mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300",
// //         disabled: isLoading,
// //       },
// //       isLoading ? "Envoi..." : "Envoyer"
// //     )
// //   );
// // };

// // export default MessageInput;
// import React, { useState, useCallback } from "react";
// import Button from "@mui/material/Button";
// import  CircularProgress  from "@mui/material/CircularProgress";
// import  Card  from "@mui/material/Card";
// import  CardContent  from "@mui/material/CardContent";
// import  CardHeader  from "@mui/material/CardHeader";
// import  TextField  from "@mui/material/TextField";
// import  Typography  from "@mui/material/Typography";
// import  {CheckCircle, ErrorOutline } from "@mui/icons-material";

// const MessageInput = ({ onSubmit }) => {
//   const [message, setMessage] = useState("");
//   const [isLoading, setIsLoading] = useState(false);
//   const [analysis, setAnalysis] = useState(null);

//   const handleSubmit = useCallback(
//     async (e) => {
//       e.preventDefault();
//       setIsLoading(true);
//       try {
//         // POST request to add the message
//         const postResponse = await fetch("http://localhost:5000/add_message", {
//           method: "POST",
//           headers: {
//             "Content-Type": "application/json",
//           },
//           body: JSON.stringify({
//             message: message}),
//         });

//         if (!postResponse.ok) {
//           throw new Error("Network response was not ok for POST request");
//         }

//         const postResult = await postResponse.json();
//         console.log("Message added:", postResult);

//         // GET request to analyze the message
//         const getResponse = await fetch(
//           `http://localhost:5000/message/${encodeURIComponent(message)}`
//         );

//         if (!getResponse.ok) {
//           throw new Error("Network response was not ok for GET request");
//         }

//         const analysisResult = await getResponse.json();
//         console.log("Message analysis:", analysisResult);
//         setAnalysis(analysisResult);

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
//  return (
//     <div className="max-w-2xl mx-auto p-4">
//       <Card className="mb-6">
//         <CardHeader>
//           <Typography variant="h6">Analyseur de Messages</Typography>
//         </CardHeader>
//         <CardContent>
//           <form onSubmit={handleSubmit} className="space-y-4">
//             <TextField
//               label="Tapez votre message ici"
//               type="text"
//               value={message}
//               onChange={(e) => setMessage(e.target.value)}
//               fullWidth
//               variant="outlined"
//               disabled={isLoading}
//             />
//             <Button
//               type="submit"
//               fullWidth
//               variant="contained"
//               color="primary"
//               disabled={isLoading}
//             >
//               {isLoading ? <CircularProgress size={24} /> : "Envoyer"}
//             </Button>
//           </form>
//         </CardContent>
//       </Card>

//       {analysis && (
//         <Card>
//           <CardHeader>
//             <Typography variant="h6">Analyse du message</Typography>
//           </CardHeader>
//           <CardContent>
//             <div className="space-y-4">
//               <Typography>
//                 <strong>Message :</strong> {analysis.message}
//               </Typography>
//               <div>
//                 <strong>Résultats des algorithmes :</strong>
//                 {Object.entries(analysis.results).map(
//                   ([algorithm, result], index) => (
//                     <Card key={index} className="mb-4">
//                       <CardHeader>
//                         <Typography variant="h6">
//                           {algorithm}
//                         </Typography>
//                       </CardHeader>
//                       <CardContent>
//                         <p>
//                           <strong>Type de cyberharcèlement :</strong>{" "}
//                           {result.type_de_cyberharcèlement}
//                         </p>
//                         <p>
//                           <strong>Probabilité maximale :</strong>{" "}
//                           {(result.probabilité_maximale * 100).toFixed(2)}%
//                         </p>
//                         <div>
//                           <strong>Probabilités par classe :</strong>
//                           <ul className="list-disc list-inside pl-4 mt-1">
//                             {Object.entries(result.probabilités_par_classe).map(
//                               ([key, value]) => (
//                                 <li key={key}>
//                                   {key} : {(value * 100).toFixed(2)}%
//                                 </li>
//                               )
//                             )}
//                           </ul>
//                         </div>
//                       </CardContent>
//                     </Card>
//                   )
//                 )}
//               </div>
//             </div>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   );
// };

// export default MessageInput;

import React, { useState, useCallback } from "react";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardHeader from "@mui/material/CardHeader";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";

const MessageInput = ({ onSubmit }) => {
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setIsLoading(true);
      try {
        const postResponse = await fetch("http://localhost:5000/add_message", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
          }),
        });

        if (!postResponse.ok) {
          throw new Error("Network response was not ok for POST request");
        }

        const postResult = await postResponse.json();
        console.log("Message added:", postResult);

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

  const renderAlgorithmCard = (algorithmName) => {
    const result = analysis?.results?.[algorithmName];

    return (
      <Card className="h-full">
        <CardHeader>
          <Typography variant="h6" className="text-center font-bold">
            {algorithmName}
          </Typography>
        </CardHeader>
        <CardContent>
          {!analysis ? (
            <Typography className="text-center text-gray-500">
              En attente d'un message à analyser...
            </Typography>
          ) : (
            <div className="space-y-4">
              <Typography>
                <strong>Type de cyberharcèlement :</strong>{" "}
                {result?.type_de_cyberharcèlement || "Non disponible"}
              </Typography>
              <Typography>
                <strong>Probabilité maximale :</strong>{" "}
                {result
                  ? `${(result.probabilité_maximale * 100).toFixed(2)}%`
                  : "Non disponible"}
              </Typography>
              <div>
                <Typography>
                  <strong>Probabilités par classe :</strong>
                </Typography>
                <ul className="list-none pl-4 mt-2">
                  {result ? (
                    Object.entries(result.probabilités_par_classe).map(
                      ([key, value]) => (
                        <li key={key} className="mb-2">
                          <div className="flex justify-between items-center">
                            <span>{key}:</span>
                            <span>{(value * 100).toFixed(2)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div
                              className="bg-blue-600 h-2.5 rounded-full"
                              style={{ width: `${value * 100}%` }}
                            ></div>
                          </div>
                        </li>
                      )
                    )
                  ) : (
                    <Typography className="text-gray-500">
                      Données non disponibles
                    </Typography>
                  )}
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <Card className="mb-6">
        <CardHeader>
          <Typography variant="h5" className="text-center">
            Analyseur de Messages
          </Typography>
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
              className="mt-4"
            >
              {isLoading ? <CircularProgress size={24} /> : "Envoyer"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {analysis && (
        <Typography variant="h6" className="mb-4 text-center">
          Message analysé : {analysis.message}
        </Typography>
      )}

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          {renderAlgorithmCard("Logistic Regression")}
        </Grid>
        <Grid item xs={12} md={6}>
          {renderAlgorithmCard("SVM")}
        </Grid>
      </Grid>
    </div>
  );
};

export default MessageInput;