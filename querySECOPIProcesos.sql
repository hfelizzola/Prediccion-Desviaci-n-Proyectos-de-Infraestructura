SELECT 
        uid,
        nombre_entidad AS nombre_de_la_entidad, --already updated
        nit_de_la_entidad,
        departamento_entidad,
        orden_entidad AS clasificacion_entidad,
        UPPER(nivel_entidad) AS orden,
        modalidad_de_contratacion AS modalidad, --already updated
        UPPER(tipo_de_contrato) AS tipo_contrato,
        UPPER(causal_de_otras_formas_de) AS causal_contratacion_directa,
        objeto_a_contratar,
        UPPER(detalle_del_objeto_a_contratar) AS detalle_objeto,
        cuantia_proceso,
        cuantia_contrato,
        valor_total_de_adiciones,
        valor_contrato_con_adiciones,
        anno_firma_contrato AS anno_firma, --already updated (anno_firma_del_contrato)
        fecha_de_firma_del_contrato AS fecha_firma,
        fecha_ini_ejec_contrato,
        plazo_de_ejec_del_contrato,
        rango_de_ejec_del_contrato,
        tiempo_adiciones_en_dias,
        tiempo_adiciones_en_meses,
        fecha_fin_ejec_contrato,
        id_adjudicacion,
        ruta_proceso_en_secop_i as urlproceso,
        identificacion_del_contratista,
        es_postconflicto AS espostconflicto,
        es_mipyme AS es_pyme,
        dpto_y_muni_contratista,
        municipio_entidad,
        nom_razon_social_contratista,
        estado_del_proceso
WHERE
        anno_firma IS NOT NULL
        AND anno_firma NOT LIKE 'Sin Firma'
        AND fecha_firma IS NOT NULL
        AND detalle_objeto IS NOT NULL
        AND detalle_objeto NOT LIKE 'No Definido'
        AND tipo_contrato = 'OBRA'
        AND estado_del_proceso IN ('LIQUIDADO')
        AND anno_firma IN ('2014','2015','2016','2017','2018','2019','2020','2021','2022','2023')
        --AND anno_firma IN ('2018')
        AND cuantia_proceso > 50000000
        AND cuantia_contrato > 50000000
        AND causal_contratacion_directa NOT IN ('CONTRATOS INTERADMINISTRATIVOS (LITERAL C)', 
                                                --'NO DEFINIDO',
                                                --'PRESTACION DE SERVICIOS PROFESIONALES Y DE APOYO A LA GESTION (LITERAL H)',
                                                --'URGENCIA MANIFIESTA (LITERAL A)',
                                                'CONTRATACION DE EMPRESTITOS (LITERAL B)',
                                                'CONTRATOS PARA EL DESARROLLO DE ACTIVIDADES CIENTIFICAS Y TECNOLOGICAS (LITERAL E)'
                                                --'LA CONTRATACION DE MENOR CUANTIA (LITERAL B)'
                                                )
        --AND nombre_regimen_de_contratacion != 'REGIMEN ESPECIAL' --already updated (regimen_de_contratacion)
        --AND tipo_de_proceso IN ('LICITACION PUBLICA','LICITACION OBRA PUBLICA')
        
        --QUITAR CONVENIOS INTERADMINISTRATIVOS
        AND detalle_objeto NOT LIKE '%AUNAR%'
        AND detalle_objeto NOT LIKE '%ANUAR%'
        AND detalle_objeto NOT LIKE '%AUNNAR%'
        AND detalle_objeto NOT LIKE '%UNAR%ESFUERZO%'
        AND detalle_objeto NOT LIKE '%UNNAR%ESFUERZO%'
        AND detalle_objeto NOT LIKE '%AUNAR%ESPUERZO%'
        AND detalle_objeto NOT LIKE '%CONVE%IO%'
        AND detalle_objeto NOT LIKE '%INTERADMINISTRATIVO%'
        AND detalle_objeto NOT LIKE '%MANTENIMIENTO%RUTINARIO%'
        AND detalle_objeto NOT LIKE '%COMPLEMENTA%ESFUERZOS%INSTITUCIONALES%'
        AND detalle_objeto NOT LIKE '%UNI%ESFUERZOS%'
        
        --PRESTACIÓN DE SERVICIOS, ACTIVIDADES DE GESTIÓN Y CONSULTORIA
        AND detalle_objeto NOT LIKE '%PRESTA%SERVICIO%'
        AND detalle_objeto NOT LIKE '%ADMINISTRACI%N%'
        AND detalle_objeto NOT LIKE '%ELABORACI%N%MANUAL%'
        AND detalle_objeto NOT LIKE '%CONSULTOR%A%'
        AND detalle_objeto NOT LIKE '%ARTICULAR%ESTRATEGIAS%'
        
        --OPERACIONES FINANCIERAS
        AND detalle_objeto NOT LIKE '%COFINANCIA%'
        
        -- DISEÑO, ESTUDIOS E INTERVENTORIA
        AND detalle_objeto NOT LIKE '%INTERVENTOR%A%' 
        AND detalle_objeto NOT LIKE '%CONSULTOR%A%' 
        AND detalle_objeto NOT LIKE '%ESTUDIO%'
        AND detalle_objeto NOT LIKE '%INTERVENTOR%A%'
        AND detalle_objeto NOT LIKE '%DISEÑO%'
        AND detalle_objeto NOT LIKE '%DISENO%' 
        
        -- OBRAS COMPLEMENTARIAS Y OTROS TIPOS DE OBRAS
        --AND detalle_objeto NOT LIKE '%ACUEDUCTO%'
        AND detalle_objeto NOT LIKE '%AFECTACIONES%'
        AND detalle_objeto NOT LIKE '%ALQUILER%'
        --AND detalle_objeto NOT LIKE '%ALCANTARILLA%'
        AND detalle_objeto NOT LIKE '%ANDEN%'
        --AND detalle_objeto NOT LIKE '%AULAS%'
        AND detalle_objeto NOT LIKE '%BARANDAS%'
        --AND detalle_objeto NOT LIKE '%BOX%C%ULVERT%'
        --AND detalle_objeto NOT LIKE '%CALLE%'
        --AND detalle_objeto NOT LIKE '%CARRERA%'
        --AND detalle_objeto NOT LIKE '%CARRETERA%'
        --AND detalle_objeto NOT LIKE '%CICLOV%AS%'
        --AND detalle_objeto NOT LIKE '%CICLORUTA%'
        --AND detalle_objeto NOT LIKE '%COLEGIO%'
        --AND detalle_objeto NOT LIKE '%CONSERVACI%N%PUENTE%'
        AND detalle_objeto NOT LIKE '%DEMOLICI%N%'
        --AND detalle_objeto NOT LIKE '%DESASTRE%'
        --AND detalle_objeto NOT LIKE '%DESMONTE%LIMPIEZA%'
        AND detalle_objeto NOT LIKE '%DESLIZAMIENTO%TALUD%'
        AND detalle_objeto NOT LIKE '%EJERCITO%'
        AND detalle_objeto NOT LIKE '%EMERGENCIA%'
        AND detalle_objeto NOT LIKE '%INSTITUCI%N%EDUCATIVA%'
        AND detalle_objeto NOT LIKE '%INVENTARIO%'
        AND detalle_objeto NOT LIKE '%LIMPIEZA%'
        AND detalle_objeto NOT LIKE '%REMOCION%'
        AND detalle_objeto NOT LIKE '%MANO%DE%OBRA%'
        AND detalle_objeto NOT LIKE '%MANTENIMIENTO%'
        AND detalle_objeto NOT LIKE '%MANTENIMIENTO%T%NEL%'
        --AND detalle_objeto NOT LIKE '%MEZCLA%ASF%LTICA%'
        AND detalle_objeto NOT LIKE '%MITIGACI%N%'
        --AND detalle_objeto NOT LIKE '%MUROS%'
        AND detalle_objeto NOT LIKE '%OLA%INVERNAL%'
        --AND detalle_objeto NOT LIKE '%PARQUE%'
        --AND detalle_objeto NOT LIKE '%PISTA%'
        --AND detalle_objeto NOT LIKE '%PUENTE%PEATONAL%'
        --AND detalle_objeto NOT LIKE '%PUENTES%COLGANTES%'
        --AND detalle_objeto NOT LIKE '%REHABILITACI%N%CONSERVACI%N%PUENTE%'
        --AND detalle_objeto NOT LIKE '%RECUPERACI%O%'
        AND detalle_objeto NOT LIKE '%REDUCTORES%VELOCIDAD%'
        AND detalle_objeto NOT LIKE '%REMOCION%'
        --AND detalle_objeto NOT LIKE '%RESTAURACI%N%ESTACI%N%F%RREA%'
        AND detalle_objeto NOT LIKE '%RESIDUOS%S%LIDOS%'
        --AND detalle_objeto NOT LIKE '%ROCER%A%'
        --AND detalle_objeto NOT LIKE '%ROSER%A%'
        AND detalle_objeto NOT LIKE '%SERVICIO%ALQUILER%'
        AND detalle_objeto NOT LIKE '%SEMAFORIZACI%N%'
        AND detalle_objeto NOT LIKE '%SEÑALIZACI%N%'
        AND detalle_objeto NOT LIKE '%SE%ALIZACION%'
        AND detalle_objeto NOT LIKE '%SENALIZACI%N%'
        AND detalle_objeto NOT LIKE '%SUMINISTRO%'
        --AND detalle_objeto NOT LIKE '%URBAN%'
        
    
        --AND (detalle_objeto NOT LIKE '%CONSTRUCCI%N%PUENTE%' OR detalle_objeto LIKE '%PUENTE%NACIONAL%')
        --AND (detalle_objeto NOT LIKE '%SECUNDARIA%' OR detalle_objeto NOT LIKE '%TERCIARIA%')
        --CONTAIN FILTERS
        --AND (detalle_objeto LIKE '%VIA%' OR detalle_objeto LIKE '%VIAS%' OR detalle_objeto LIKE '%VIAL%' 
        --AND (detalle_objeto LIKE '%CONSTRUCCI%N%HUELLA%' OR detalle_objeto LIKE '%CONSTRUCCI%N%PLACA%' 
        --        OR detalle_objeto LIKE '%PLACA%HUELLA%' OR detalle_objeto LIKE '%VEREDA%' OR detalle_objeto LIKE '%RURAL%'
        --        OR detalle_objeto LIKE '%RURALES%' OR detalle_objeto LIKE '%VEREDAL%' OR detalle_objeto LIKE '%VEREDALES%'
          --      OR detalle_objeto LIKE '%TERCIARIA%' OR detalle_objeto LIKE '%TERCEARIA%' OR detalle_objeto LIKE '%TERCI%')
LIMIT
        100000