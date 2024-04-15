import React, { createContext, useEffect, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import "./styles/index.scss";

import Navbar from "./components/NavBar";
import SideBar from "./components/SideBar";
import AddVital from "./pages/AddVital";
import Auth from "./pages/Auth";
import Home from "./pages/Home";
import ManageAccount from "./pages/ManageAccount";
import AddPatient from "./pages/nurse/AddPatient";
import Consultation from "./pages/nurse/Consultation";
import NurseDashboard from "./pages/nurse/NurseDashboard";
import Patient from "./pages/nurse/Patient";
import PatientDashboard from "./pages/patient/PatientDashboard";
import DiagnosisApp from "./components/Diagnosis";
import useAuth from "./utils/useAuth";

const endpoints = ["/", "/login", "/register"];

export const AppContext = createContext(null);

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const [isApp, setIsApp] = useState(false);

  const location = useLocation();

  useEffect(() => {
    setIsApp(!endpoints.includes(location.pathname));
  }, [location.pathname]);

  return (
    <AppContext.Provider value={{ isLoggedIn, setIsLoggedIn, isApp }}>
      <Navbar />
      <main className="main-container">
        {isApp && <SideBar />}
        <section className="page-container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Auth />} />
            <Route path="/register" element={<Auth />} />
            <Route path="/patient" element={<PatientDashboard />} />
            <Route path="/nurse" element={<NurseDashboard />} />
            <Route path="/manage-account" element={<ManageAccount />} />
            <Route path="/add-vital" element={<AddVital />} />
            <Route path="/nurse/add-patient" element={<AddPatient />} />
            <Route path="/nurse/patient/:id" element={<Patient />} />
            <Route path="/nurse/consultation" element={<Consultation />} />
            {/* Change the path as you see fit */}
            <Route path="/diagnosis" element={<DiagnosisApp />} />
          </Routes>
        </section>
      </main>
    </AppContext.Provider>
  );
}

export default App;
