import React, { useEffect, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import "./styles/index.scss";

import Navbar from "./components/NavBar";
import SideBar from "./components/SideBar";
import Auth from "./pages/Auth";
import Books from "./pages/Books";
import Borrow from "./pages/Borrow";
import Home from "./pages/Home";
import AddPatient from "./pages/nurse-dashboard/AddPatient";
// import Members from "./pages/Members";
import NurseDashboard from "./pages/NurseDashboard";
import Publishers from "./pages/Publishers";
import Patient from "./pages/nurse-dashboard/Patient";

const endpoints = ["/", "/login", "/register"];

function App() {
  const [isApp, setIsApp] = useState(false);

  const location = useLocation();

  useEffect(() => {
    setIsApp(!endpoints.includes(location.pathname));
  }, [location.pathname]);

  return (
    <>
      <Navbar />
      <main className="main-container">
        {isApp && <SideBar />}
        <section className="page-container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Auth />} />
            <Route path="/register" element={<Auth />} />
            <Route path="/app" element={<Books />} />
            <Route path="/app/nurse" element={<NurseDashboard />} />
            <Route path="/add-patient" element={<AddPatient />} />
            <Route path="/patient/:id" element={<Patient />} />
          </Routes>
        </section>
      </main>
    </>
  );
}

export default App;
