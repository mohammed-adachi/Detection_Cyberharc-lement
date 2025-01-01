import React from "react";

const Settings = () => {
  return React.createElement(
    "div",
    { className: "p-6" },
    React.createElement(
      "h2",
      { className: "text-3xl font-bold mb-6" },
      "Settings"
    ),
    React.createElement(
      "div",
      { className: "space-y-6" },
      React.createElement(
        "div",
        null,
        React.createElement(
          "h3",
          { className: "text-xl font-semibold mb-2" },
          "Detection Threshold"
        ),
        React.createElement("input", {
          type: "range",
          min: "0",
          max: "100",
          className: "w-full",
        })
      ),
      React.createElement(
        "div",
        null,
        React.createElement(
          "h3",
          { className: "text-xl font-semibold mb-2" },
          "Notification Settings"
        ),
        React.createElement(
          "label",
          { className: "flex items-center" },
          React.createElement("input", { type: "checkbox", className: "mr-2" }),
          "Email notifications"
        ),
        React.createElement(
          "label",
          { className: "flex items-center mt-2" },
          React.createElement("input", { type: "checkbox", className: "mr-2" }),
          "Push notifications"
        )
      ),
      React.createElement(
        "div",
        null,
        React.createElement(
          "h3",
          { className: "text-xl font-semibold mb-2" },
          "Filtering Options"
        ),
        React.createElement(
          "label",
          { className: "flex items-center" },
          React.createElement("input", { type: "checkbox", className: "mr-2" }),
          "Auto-filter high-risk messages"
        ),
        React.createElement(
          "label",
          { className: "flex items-center mt-2" },
          React.createElement("input", { type: "checkbox", className: "mr-2" }),
          "Show message context"
        )
      )
    )
  );
};

export default Settings;
