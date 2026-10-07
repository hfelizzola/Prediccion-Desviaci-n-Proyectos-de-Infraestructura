SELECT
    id_del_portafolio AS id_proceso,
    precio_base AS cuantia_proceso,
    valor_total_adjudicacion AS cuantia_contrato,
    duracion AS plazo_de_ejec_del_contrato,
    unidad_de_duracion AS rango_de_ejec_del_contrato_y,
    departamento_proveedor,
    ciudad_proveedor,
    id_adjudicacion,
    fecha_adjudicacion,
    adjudicado,
    estado_resumen
WHERE
    id_proceso IN ({list_id_proceso})
    AND estado_resumen = 'Adjudicado'
    AND valor_total_adjudicacion IS NOT NULL
    AND fecha_adjudicacion	IS NOT NULL
    AND adjudicado = 'Si'
    AND cuantia_proceso > 0
    AND cuantia_contrato > 0
    AND id_adjudicacion NOT IN ('No Definido','No Adjudicado')
