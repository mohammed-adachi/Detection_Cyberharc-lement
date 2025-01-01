import React from "react";

const Dashboard = () => {
  return (
    <div className="p-6">
      <h2 className="text-3xl font-bold mb-6">Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Card 1 */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Total Messages</h3>
          <p className="text-3xl font-bold">1,234</p>
        </div>

        {/* Card 2 */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Harassment Detected</h3>
          <p className="text-3xl font-bold text-red-600">56</p>
        </div>

        {/* Card 3 */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Detection Rate</h3>
          <p className="text-3xl font-bold text-green-600">4.5%</p>
        </div>

        {/* Card 4 */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Alerts</h3>
          <p className="text-3xl font-bold text-yellow-600">3</p>
        </div>
      </div>

      {/* Recent Alerts Section */}
      <div className="mt-8 bg-white p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold mb-4">Recent Alerts</h3>
        <ul className="space-y-2">
          <li className="flex items-center text-red-600">
            <span className="mr-2">•</span>
            High-risk message detected in Chat Room #5
          </li>
          <li className="flex items-center text-yellow-600">
            <span className="mr-2">•</span>
            Unusual activity spike in Forum Thread #123
          </li>
          <li className="flex items-center text-yellow-600">
            <span className="mr-2">•</span>
            Multiple reports against User ID: 789
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;
