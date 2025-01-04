import React, { useState } from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`algorithm-tabpanel-${index}`}
      aria-labelledby={`algorithm-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Dashboard = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const metrics = {
    logisticRegression: {
      title: "Régression Logistique",
      description:
        "La régression logistique est un algorithme de classification qui estime la probabilité qu'une instance appartienne à une classe. Dans le contexte du cyberharcèlement, elle analyse les caractéristiques du texte pour prédire la probabilité qu'un message soit harcelant.",
      detailedReport: {
        precision: [0.95, 0.88],
        recall: [0.93, 0.92],
        f1Score: [0.94, 0.9],
        support: [1980, 1183],
      },
      globalMetrics: {
        precision: 0.928,
        recall: 0.926,
        f1Score: 0.927,
        accuracy: 0.926,
        logLoss: 0.2426,
      },
    },
    svm: {
      title: "SVM (Support Vector Machine)",
      description:
        "Le SVM est un algorithme qui trouve l'hyperplan optimal séparant les classes dans un espace multidimensionnel. Il est particulièrement efficace pour la classification de texte car il gère bien les espaces de grande dimension.",
      detailedReport: {
        precision: [0.96, 0.89],
        recall: [0.93, 0.93],
        f1Score: [0.94, 0.91],
        support: [1980, 1183],
      },
      globalMetrics: {
        precision: 0.933,
        recall: 0.931,
        f1Score: 0.932,
        accuracy: 0.931,
        logLoss: 0.1964,
      },
    },
  };

  const renderMetricsCard = (algorithm) => (
    <div style={{ marginTop: "24px" }}>
      <Card>
        <CardContent>
          <Typography variant="h5" component="h3" gutterBottom>
            {algorithm.title}
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            {algorithm.description}
          </Typography>

          <Box sx={{ my: 4 }}>
            <Typography variant="h6" gutterBottom>
              Rapport de classification détaillé
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Classe</TableCell>
                    <TableCell>Précision</TableCell>
                    <TableCell>Rappel</TableCell>
                    <TableCell>F1-score</TableCell>
                    <TableCell>Support</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Non-harcèlement (-1.0)</TableCell>
                    <TableCell>
                      {algorithm.detailedReport.precision[0]}
                    </TableCell>
                    <TableCell>{algorithm.detailedReport.recall[0]}</TableCell>
                    <TableCell>{algorithm.detailedReport.f1Score[0]}</TableCell>
                    <TableCell>{algorithm.detailedReport.support[0]}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Harcèlement (0.0)</TableCell>
                    <TableCell>
                      {algorithm.detailedReport.precision[1]}
                    </TableCell>
                    <TableCell>{algorithm.detailedReport.recall[1]}</TableCell>
                    <TableCell>{algorithm.detailedReport.f1Score[1]}</TableCell>
                    <TableCell>{algorithm.detailedReport.support[1]}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Box>

          <Box sx={{ my: 4 }}>
            <Typography variant="h6" gutterBottom>
              Métriques globales
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: 2,
              }}
            >
              {Object.entries(algorithm.globalMetrics).map(([key, value]) => (
                <Card key={key} variant="outlined">
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </Typography>
                    <Typography variant="h5">
                      {key === "logLoss"
                        ? value.toFixed(4)
                        : (value * 100).toFixed(1) + "%"}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Box>

          <Box sx={{ my: 4 }}>
            <Typography variant="h6" gutterBottom>
              Matrice de Confusion
            </Typography>
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <img
                src={`/api/placeholder/500/400`}
                alt={`Matrice de confusion pour ${algorithm.title}`}
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
                }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </div>
  );

  return (
    <Box sx={{ p: 3, maxWidth: "1200px", margin: "0 auto" }}>
      <Typography variant="h4" component="h2" gutterBottom>
        Dashboard d'Analyse
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs value={tabValue} onChange={handleTabChange} centered>
          <Tab label="Régression Logistique" />
          <Tab label="SVM" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        {renderMetricsCard(metrics.logisticRegression)}
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        {renderMetricsCard(metrics.svm)}
      </TabPanel>
    </Box>
  );
};

export default Dashboard;
