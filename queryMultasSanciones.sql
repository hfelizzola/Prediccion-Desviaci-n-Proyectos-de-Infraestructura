SELECT 
    numero_de_contrato AS penaltyId,
    nombre_entidad AS buyerName,
    nit_entidad AS buyerId,
    valor_sancion AS penaltyValue,
    fecha_de_firmeza AS penaltyDate
WHERE  
    buyerId IN ({NIT_ENTIDAD_LIST})