import React from "react";
import { Link } from "react-router";
import App from "../App";

const Sidebar = () => {
  return React.createElement(
    "nav",
    { className: "w-64 bg-white shadow-lg" },
    React.createElement(
      "div",
      { className: "p-4" },
      React.createElement(
        "h1",
        { className: "text-2xl font-bold text-gray-800" },
        "Harassment Detection"
      ),
      React.createElement(
        "ul",
        { className: "mt-6 space-y-2" },
        React.createElement(
          "li",
          null,
          React.createElement(
            Link,
            {
              to: "/",
              className:
                "block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded",
            },
            "Dashboard"
          )
        ),
        React.createElement(
          "li",
          null,
          React.createElement(
            Link,
            {
              to: "/messages",
              className:
                "block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded",
            },
            "Messages"
          )
        ),
        React.createElement(
          "li",
          null,
          React.createElement(
            Link,
            {
              to: "/settings",
              className:
                "block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded",
            },
            "Settings"
          )
        )
      )
    )
  );
};

export default Sidebar;
