import React from "react";
import { BrowserRouter , Route, Routes } from "react-router";
import Dashboard from "./components/Dashboard";
import Messages from "./components/MessageView";
import Settings from "./components/Settings";
import Sidebar from "./components/Sidebar";
import "./App.css";
import "./index.css";
const App = () => {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/messages" element={<Messages />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
};

export default App;
