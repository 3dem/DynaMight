#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:18:02 2023

@author: schwab
"""


from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
import torch
from .utils_new import*
import mrcfile
#from emda import emda_methods


def pdb2graphs(name):

    O_e = 8
    N_e = 7
    C_e = 6
    S_e = 16
    P_e = 15

    pdb = PDBParser(PERMISSIVE=1)
    cif = MMCIFParser()
    stop = 0
    gr = torch.empty(0)
    gr_DNB = torch.empty(0)
    gr_CB1 = []
    gr_CB2 = []
    gr_SC1 = []
    gr_SC2 = []
    gr_SC_SC1 = []
    gr_SC_SC2 = []
    gr_DNS1 = []
    gr_DNS2 = []
    gr_DNS_SC1 = []
    gr_DNS_SC2 = []
    total_length = 0
    SC_nr = 0
    DNS_nr = 0
    total_gr = []
    total_amp = []
    MC_c_id = 1
    CB_c_id = 0
    SC_c_id = 0
    DNB_c_id = 0
    DNS_c_id = 0
    name = str(name)

    if name.endswith('.pdb'):
        model = pdb.get_structure('model', name)
    elif name.endswith('.cif'):
        model = cif.get_structure('model', name)

    # loop through all chains

    for chain in model.get_chains():
        coords = []
        MC_coords = []  # mainchain coordinates
        MC_amps = []
        CB_coords = []  # Cbeta coordinates
        CB_amps = []
        SC_coords = []  # sidechain coordinates
        SC_amps = []
        DNB_coords = []
        DNB_amps = []
        DNS_coords = []
        DNS_amps = []
        amp = []
        add_amp = []
        types = []
        direction = []
        residues = chain.get_residues()
        for res in residues:
            rname = res.get_resname()
            if res.get_resname() in ['PHE', 'ASP', 'ASN', 'MET', 'GLU', 'GLN', 'LYS', 'HIS', 'ARG', 'ILE', 'LEU', 'TYR', 'TRP', 'PRO']:
                cm = res.center_of_mass()
                rname = res.get_resname()
                # Coarse Graining of Protein
                try:
                    Npos = res['N'].get_coord()
                    #Npos = res['N'].get_coord()

                except:
                    print('no N position', chain)
                try:
                    center_coord = (Cpos+Opos+Npos)/3
                    MC_coords.append(center_coord)
                    MC_amps.append(C_e+O_e+N_e)
                    MC_c_id += 1
                except:
                    print('no previous positions')
            try:
                Cpos = res['C'].get_coord()
                Opos = res['O'].get_coord()
            except:
                pass
            if 'PHE' in rname:
                try:
                    add_center_coord = (res['CD1'].get_coord() + res['CD2'].get_coord(
                    ) + res['CE1'].get_coord() + res['CE2'].get_coord() + res['CZ'].get_coord())/5
                    SC_coords.append(add_center_coord)
                    SC_amps.append(5*C_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1

                except:
                    pass

            if 'ASP' in rname:
                try:
                    add_center_coord = (
                        res['OD1'].get_coord() + res['OD2'].get_coord())/2
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*O_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'ASN' in rname:
                try:
                    add_center_coord = (
                        res['OD1'].get_coord() + res['ND2'].get_coord())/2
                    SC_coords.append(add_center_coord)
                    SC_amps.append(O_e+N_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'MET' in rname:
                try:
                    add_center_coord = (
                        res['SD'].get_coord() + res['CE'].get_coord())/2
                    SC_coords.append(add_center_coord)
                    SC_amps.append(C_e+S_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'GLU' in rname:
                try:
                    add_center_coord = (res['CD'].get_coord(
                    ) + res['OE1'].get_coord()+res['OE2'].get_coord())/3
                    SC_coords.append(add_center_coord)
                    SC_amps.append(C_e+2*O_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'GLN' in rname:
                try:
                    add_center_coord = (res['CD'].get_coord(
                    ) + res['OE1'].get_coord()+res['NE2'].get_coord())/3
                    SC_coords.append(add_center_coord)
                    SC_amps.append(C_e+O_e+N_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    print(rname)
                    print(res.child_list)
            if 'LYS' in rname:
                try:
                    add_center_coord = (res['CD'].get_coord(
                    ) + res['CE'].get_coord()+res['NZ'].get_coord())/3
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*C_e+N_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'HIS' in rname:
                try:
                    add_center_coord = (res['ND1'].get_coord(
                    ) + res['CD2'].get_coord()+res['CE1'].get_coord()+res['NE2'].get_coord())/4
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*C_e+2*N_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'ARG' in rname:
                try:
                    add_center_coord = (res['CD'].get_coord() + res['NE'].get_coord(
                    )+res['CZ'].get_coord()+res['NH1'].get_coord()+res['NH2'].get_coord())/5
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*C_e+3*N_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    print(rname)
                    print(res.child_list)
            if 'ILE' in rname:
                try:
                    add_center_coord = (
                        res['CG2'].get_coord() + res['CD1'].get_coord())/2
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*C_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG1'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'LEU' in rname:
                try:
                    add_center_coord = (
                        res['CD1'].get_coord() + res['CD2'].get_coord())/2
                    SC_coords.append(add_center_coord)
                    SC_amps.append(2*C_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    SC_c_id += 1
                    CB_c_id += 1
                except:
                    pass

            if 'TYR' in rname:
                try:
                    add_center_coord_1 = (res['CD1'].get_coord(
                    ) + res['CD2'].get_coord()+res['CE1'].get_coord()+res['CE2'].get_coord())/4
                    add_center_coord_2 = (
                        res['CZ'].get_coord()+res['OH'].get_coord())/2
                    SC_coords.append(add_center_coord_1)
                    SC_coords.append(add_center_coord_2)
                    SC_amps.append(4*C_e)
                    SC_amps.append(C_e+O_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    gr_SC_SC1.append(SC_c_id)
                    gr_SC_SC2.append(SC_c_id+1)
                    SC_c_id += 2
                    CB_c_id += 1
                except:
                    pass

            if 'TRP' in rname:
                try:
                    add_center_coord_1 = (res['CD1'].get_coord(
                    ) + res['CD2'].get_coord()+res['CE2'].get_coord()+res['NE1'].get_coord())/4
                    add_center_coord_2 = (res['CE3'].get_coord(
                    )+res['CH2'].get_coord()+res['CZ2'].get_coord()+res['CZ3'].get_coord())/4
                    SC_coords.append(add_center_coord_1)
                    SC_coords.append(add_center_coord_2)
                    SC_amps.append(3*C_e+N_e)
                    SC_amps.append(4*C_e)
                    CBpos = (res['CB'].get_coord() +
                             res['CA'].get_coord()+res['CG'].get_coord())/3
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    gr_SC1.append(CB_c_id)
                    gr_SC2.append(SC_c_id)
                    gr_SC_SC1.append(SC_c_id)
                    gr_SC_SC2.append(SC_c_id+1)
                    SC_c_id += 2
                    CB_c_id += 1
                except:
                    pass

            if 'PRO' in rname:
                try:
                    CBpos = (res['CB'].get_coord()+res['CA'].get_coord() +
                             res['CG'].get_coord()+res['CD'].get_coord())/4
                    CB_coords.append(CBpos)
                    CB_amps.append(4*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    CB_c_id += 1
                except:
                    pass

            if rname in ['SER', 'CYS', 'SER', 'ALA', 'GLY', 'VAL', 'THR']:
                try:
                    CBpos = res['CB'].get_coord()
                    CB_coords.append(CBpos)
                    CB_amps.append(3*C_e)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-2)
                    gr_CB1.append(CB_c_id)
                    gr_CB2.append(MC_c_id-1)
                    CB_c_id += 1
                except:
                    pass
            # Coarse Graining of DNA
            if 'DA' or 'A' in rname:
                try:
                    p_coord = res['P'].get_coord()
                    s_coord = (res["C3'"].get_coord()+res["C2'"].get_coord()+res["C1'"].get_coord(
                    )+res["C4'"].get_coord()+res["O4'"].get_coord()+res["C5'"].get_coord())/6
                    b_coord1 = (res['N9'].get_coord()+res['C8'].get_coord() +
                                res['N7'].get_coord()+res['C5'].get_coord()+res['C4'].get_coord())/5
                    b_coord2 = (res['C6'].get_coord()+res['N6'].get_coord() +
                                res['N1'].get_coord()+res['C2'].get_coord()+res['N3'].get_coord())/5
                    DNB_coords.append(p_coord)
                    DNB_coords.append(s_coord)
                    DNS_coords.append(b_coord1)
                    DNS_coords.append(b_coord2)
                    DNB_amps.append(P_e+4*O_e)
                    DNB_amps.append(5*C_e+O_e)
                    DNS_amps.append(3*C_e+2*N_e)
                    DNS_amps.append(2*C_e+3*N_e)
                    gr_DNS1.append(DNB_c_id+1)
                    gr_DNS2.append(DNS_c_id)
                    gr_DNS_SC1.append(DNS_c_id)
                    gr_DNS_SC2.append(DNS_c_id+1)
                    DNB_c_id += 2
                    DNS_c_id += 2
                except:
                    pass
            if 'DC' or 'C' in rname:
                try:
                    p_coord = res['P'].get_coord()
                    s_coord = (res["C3'"].get_coord()+res["C2'"].get_coord()+res["C1'"].get_coord(
                    )+res["C4'"].get_coord()+res["O4'"].get_coord()+res["C5'"].get_coord())/6
                    b_coord1 = (res['N1'].get_coord(
                    )+res['C2'].get_coord()+res['O2'].get_coord()+res['N3'].get_coord())/4
                    b_coord2 = (res['C4'].get_coord(
                    )+res['N4'].get_coord()+res['C5'].get_coord()+res['C6'].get_coord())/4
                    DNB_coords.append(p_coord)
                    DNB_coords.append(s_coord)
                    DNS_coords.append(b_coord1)
                    DNS_coords.append(b_coord2)
                    DNB_amps.append(P_e+4*O_e)
                    DNB_amps.append(5*C_e+O_e)
                    DNS_amps.append(C_e+2*N_e+O_e)
                    DNS_amps.append(3*C_e+N_e)
                    gr_DNS1.append(DNB_c_id+1)
                    gr_DNS2.append(DNS_c_id)
                    gr_DNS_SC1.append(DNS_c_id)
                    gr_DNS_SC2.append(DNS_c_id+1)
                    DNB_c_id += 2
                    DNS_c_id += 2
                except:
                    pass
            if 'DG' or 'G' or 'U' in rname:
                try:
                    p_coord = res['P'].get_coord()
                    s_coord = (res["C3'"].get_coord()+res["C2'"].get_coord()+res["C1'"].get_coord(
                    )+res["C4'"].get_coord()+res["O4'"].get_coord()+res["C5'"].get_coord())/6
                    b_coord1 = (res['N9'].get_coord()+res['C8'].get_coord() +
                                res['N7'].get_coord()+res['C5'].get_coord()+res['C4'].get_coord())/5
                    b_coord2 = (res['C6'].get_coord()+res['O6'].get_coord() +
                                res['N1'].get_coord()+res['C2'].get_coord()+res['N3'].get_coord())/5
                    DNB_coords.append(p_coord)
                    DNB_coords.append(s_coord)
                    DNS_coords.append(b_coord1)
                    DNS_coords.append(b_coord2)
                    DNB_amps.append(P_e+4*O_e)
                    DNB_amps.append(5*C_e+O_e)
                    DNS_amps.append(2*N_e+3*C_e)
                    DNS_amps.append(2*C_e+2*N_e+O_e)
                    gr_DNS1.append(DNB_c_id+1)
                    gr_DNS2.append(DNS_c_id)
                    gr_DNS_SC1.append(DNS_c_id)
                    gr_DNS_SC2.append(DNS_c_id+1)
                    DNB_c_id += 2
                    DNS_c_id += 2
                except:
                    pass
            if 'DT' in rname:
                try:
                    p_coord = res['P'].get_coord()
                    s_coord = (res["C3'"].get_coord()+res["C2'"].get_coord()+res["C1'"].get_coord(
                    )+res["C4'"].get_coord()+res["O4'"].get_coord()+res["C5'"].get_coord())/6
                    b_coord1 = (res['N1'].get_coord(
                    )+res['C2'].get_coord()+res['O2'].get_coord()+res['N3'].get_coord())/4
                    b_coord2 = (res['C4'].get_coord()+res['O4'].get_coord() +
                                res['C5'].get_coord()+res['C7'].get_coord()+res['C6'].get_coord())/5
                    DNB_coords.append(p_coord)
                    DNB_coords.append(s_coord)
                    DNS_coords.append(b_coord1)
                    DNS_coords.append(b_coord2)
                    DNB_amps.append(P_e+4*O_e)
                    DNB_amps.append(5*C_e+O_e)
                    DNS_amps.append(C_e+O_e+2*N_e)
                    DNS_amps.append(4*C_e+O_e)
                    gr_DNS1.append(DNB_c_id+1)
                    gr_DNS2.append(DNS_c_id)
                    gr_DNS_SC1.append(DNS_c_id)
                    gr_DNS_SC2.append(DNS_c_id+1)
                    DNB_c_id += 2
                    DNS_c_id += 2
                except:
                    pass

        print('length is', len(DNB_coords), len(
            DNS_coords), len(MC_coords), 'of chain', chain)
        if len(gr) == 0 and len(MC_coords) > 0:
            gr = torch.stack([torch.arange(gr.shape[0], gr.shape[0]+len(MC_coords)-1),
                             torch.arange(gr.shape[0]+1, gr.shape[0]+len(MC_coords))], 1)

        elif len(MC_coords) > 1:
            #     print(gr[-1,1])
            gr_new = torch.stack([torch.arange(gr[-1, 1]+2, gr[-1, 1]+len(MC_coords)),
                                 torch.arange(gr[-1, 1]+3, gr[-1, 1]+len(MC_coords)+1)], 1)
            gr = torch.cat([gr, gr_new])
        if len(gr_DNB) == 0 and len(DNB_coords) > 0:
            gr_DNB = torch.stack([torch.arange(gr_DNB.shape[0], gr_DNB.shape[0]+len(
                DNB_coords)-1), torch.arange(gr_DNB.shape[0]+1, gr_DNB.shape[0]+len(DNB_coords))], 1)
        elif len(DNB_coords) > 0:
            gr_DNB_new = torch.stack([torch.arange(gr_DNB[-1, 1]+2, gr_DNB[-1, 1]+len(
                DNB_coords)), torch.arange(gr_DNB[-1, 1]+3, gr_DNB[-1, 1]+len(DNB_coords)+1)], 1)
            gr_DNB = torch.cat([gr_DNB, gr_DNB_new])

        # print(gr)
        MC_coords = torch.tensor(np.array(MC_coords))
        print(MC_coords)
        MC_amps = torch.tensor(np.array(MC_amps))
        CB_coords = torch.tensor(np.array(CB_coords))
        CB_amps = torch.tensor(np.array(CB_amps))
        SC_amps = torch.tensor(np.array(SC_amps))
        DNB_amps = torch.tensor(np.array(DNB_amps))
        DNS_amps = torch.tensor(np.array(DNS_amps))
        SC_coords = torch.tensor(np.array(SC_coords))
        DNB_coords = torch.tensor(np.array(DNB_coords))
        DNS_coords = torch.tensor(np.array(DNS_coords))
        CB_1_gr = torch.tensor(np.array(gr_CB1)[1:-1])
        CB_2_gr = torch.tensor(np.array(gr_CB2)[1:-1])
        SC_1_gr = torch.tensor(np.array(gr_SC1))
        SC_2_gr = torch.tensor(np.array(gr_SC2))
        SC_SC1_gr = torch.tensor(np.array(gr_SC_SC1))
        SC_SC2_gr = torch.tensor(np.array(gr_SC_SC2))
        DNS1_gr = torch.tensor(np.array(gr_DNS1))
        DNS2_gr = torch.tensor(np.array(gr_DNS2))
        DNS_SC1_gr = torch.tensor(np.array(gr_DNS_SC1))
        DNS_SC2_gr = torch.tensor(np.array(gr_DNS_SC2))
        gr_CB1 = []
        gr_CB2 = []
        gr_SC1 = []
        gr_SC2 = []
        gr_SC_SC1 = []
        gr_SC_SC2 = []
        gr_DNS1 = []
        gr_DNS2 = []
        gr_DNS_SC1 = []
        gr_DNS_SC2 = []

        try:
            total_SC_SC_gr = torch.cat(
                [total_SC_SC_gr, torch.stack([SC_SC1_gr, SC_SC2_gr])], 1)
        except:

            total_SC_SC_gr = torch.stack([SC_SC1_gr, SC_SC2_gr])

        try:
            total_DNS_gr = torch.cat(
                [total_DNS_gr, torch.stack([DNS1_gr, DNS2_gr])], 1)
            total_DNS_SC_gr = torch.cat(
                [total_DNS_SC_gr, torch.stack([DNS_SC1_gr, DNS_SC2_gr])], 1)
        except:
            total_DNS_gr = torch.stack([DNS1_gr, DNS2_gr])
            total_DNS_SC_gr = torch.stack([DNS_SC1_gr, DNS_SC2_gr])

        if total_length == 0:
            total_add_amp = add_amp
            total_MC_coords = MC_coords
            total_CB_coords = CB_coords
            total_SC_coords = SC_coords
            total_DNB_coords = DNB_coords
            total_DNS_coords = DNS_coords
            total_MC_amps = MC_amps
            total_CB_amps = CB_amps
            total_SC_amps = SC_amps
            total_DNB_amps = DNB_amps
            total_DNS_amps = DNS_amps
            total_MC_gr = gr
            total_CB_gr = torch.stack([CB_1_gr, CB_2_gr], 1)
            total_SC_gr = torch.stack([SC_1_gr, SC_2_gr], 1)
        else:
            total_MC_coords = torch.cat([total_MC_coords, MC_coords])
            total_CB_coords = torch.cat([total_CB_coords, CB_coords])
            total_SC_coords = torch.cat([total_SC_coords, SC_coords])
            total_DNB_coords = torch.cat([total_DNB_coords, DNB_coords])
            total_DNS_coords = torch.cat([total_DNS_coords, DNS_coords])
            total_MC_amps = torch.cat([total_MC_amps, MC_amps])
            total_CB_amps = torch.cat([total_CB_amps, CB_amps])
            total_SC_amps = torch.cat([total_SC_amps, SC_amps])
            total_DNB_amps = torch.cat([total_DNB_amps, DNB_amps])
            total_DNS_amps = torch.cat([total_DNS_amps, DNS_amps])
            total_MC_gr = torch.cat([total_MC_gr, gr])
            total_CB_gr = torch.cat(
                [total_CB_gr, torch.stack([CB_1_gr, CB_2_gr], 1)])
            total_SC_gr = torch.cat(
                [total_SC_gr, torch.stack([SC_1_gr, SC_2_gr], 1)])
            #total_SC_SC_gr = torch.cat([total_SC_SC_gr, torch.stack([SC_SC1_gr,SC_SC2_gr],1)])
        total_length += 1  # coords.shape[0]

    return total_MC_coords, total_CB_coords, total_SC_coords, total_DNB_coords, total_DNS_coords, total_MC_amps, total_CB_amps, total_SC_amps, total_DNB_amps, total_DNS_amps, gr, gr_DNB, total_CB_gr, total_SC_gr, total_SC_SC_gr, total_DNS_gr, total_DNS_SC_gr, total_amp


def pdb2fullgraph(name, box_size, angpix):
    MC_positions, CB_positions, SC_positions, DNB_positions, DNS_positions, MC_amps, CB_amps, SC_amps, DNB_amps, DNS_amps, gr, gr_DNB, gr_CB, gr_SC, gr_SC_SC, gr_DNS, gr_DNS_SC, amp = pdb2graphs(
        name)
    positionsa = torch.cat([MC_positions, CB_positions,
                           SC_positions, DNB_positions, DNS_positions], 0).float()
    positions = positionsa/(box_size*angpix)-0.5
    if torch.min(positions) < -0.5:  # Fix for strange pdbs (waving spike)
        positions = positions+0.5
    # concatenating graphs
    MC_graph = gr.movedim(0, 1).long()
    try:
        DNB_graph = gr_DNB.movedim(
            0, 1)+MC_positions.shape[0]+CB_positions.shape[0]+SC_positions.shape[0]
        DNB_graph = DNB_graph.long()
    except:
        DNB_graph = torch.empty_like(gr_DNS)
    CB_graph = gr_CB.movedim(0, 1)
    CB_graph[0, :] += MC_positions.shape[0]
    CB_graph = CB_graph.long()
    SC_graph = gr_SC.movedim(0, 1)
    SC_graph[0, :] += MC_positions.shape[0]
    SC_graph[1, :] += MC_positions.shape[0]
    SC_graph[1, :] += CB_positions.shape[0]
    SC_graph = SC_graph.long()
    SC_SC_graph = gr_SC_SC
    SC_SC_graph += MC_positions.shape[0]+CB_positions.shape[0]
    SC_SC_graph = SC_SC_graph.long()
    DNS_graph = gr_DNS.long()
    DNS_graph[0, :] += MC_positions.shape[0] + \
        CB_positions.shape[0]+SC_positions.shape[0]
    DNS_graph[1, :] += MC_positions.shape[0]+CB_positions.shape[0] + \
        SC_positions.shape[0]+DNB_positions.shape[0]
    DNS_SC_graph = gr_DNS_SC.long()
    DNS_SC_graph[0, :] += MC_positions.shape[0] + \
        CB_positions.shape[0]+SC_positions.shape[0]+DNB_positions.shape[0]
    DNS_SC_graph[1, :] += MC_positions.shape[0] + \
        CB_positions.shape[0]+SC_positions.shape[0]+DNB_positions.shape[0]
    graph = torch.cat([MC_graph, CB_graph, SC_graph,
                      SC_SC_graph, DNB_graph, DNS_graph, DNS_SC_graph], 1)
    graph = graph.long()
    # computing violating inds
    vinds = torch.linalg.norm(
        positionsa[graph[0]]-positionsa[graph[1]], axis=1) < 5
    graph = graph[:, vinds == 1]

    # concatenation amplitudes
    amp = torch.cat([MC_amps, CB_amps, SC_amps, DNB_amps, DNS_amps], 0).float()

    return positions, graph, amp


def optimize_coarsegraining(name, box_size, ang_pix, device, outdir, n_classes=1, free_gaussians=0, N_iter=50, resolution=3, ini_model=None):
    if type(name) == list:
        nn = 0
        for n in name:
            if nn == 0:
                positions, gr, amp = pdb2fullgraph(n, box_size, ang_pix)
                nn += 1
            else:
                pos_add, gr_add, amp_add = pdb2fullgraph(n, box_size, ang_pix)
                gr_add += len(positions)
                positions = torch.cat([positions, pos_add], 0)
                print(gr.shape)
                gr = torch.cat([gr, gr_add], 1)
                amp = torch.cat([amp, amp_add], 0)
    else:
        positions, gr, amp = pdb2fullgraph(name, box_size, ang_pix)

    return positions, gr, amp
